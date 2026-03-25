"""
PPOTradingAgent and SACTradingAgent: Stable-Baselines3 wrappers for
training and evaluating RL agents on the TradingEnvironment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _compute_sharpe(returns: np.ndarray, risk_free: float = 0.05, periods: int = 252) -> float:
    """Annualised Sharpe ratio from a returns array."""
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free / periods
    excess = returns - daily_rf
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods))


def _compute_max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12)
    return float(np.min(dd))


def _evaluate_agent(
    model: Union[PPO, SAC],
    env: gym.Env,
    n_episodes: int = 100,
) -> Dict[str, float]:
    """
    Roll out n_episodes in env and compute performance metrics.

    Returns
    -------
    dict with keys: mean_return, std_return, sharpe, max_drawdown, win_rate,
                    mean_episode_length
    """
    all_returns: list = []
    all_equity: list = []
    all_lengths: list = []
    wins = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_returns: list = []
        equity_curve = [1.0]
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Use info for a cleaner equity curve when available
            if isinstance(info, dict) and "portfolio_value" in info:
                equity_curve.append(info["portfolio_value"])
            episode_returns.append(float(reward))
            step += 1

        ep_return = float(np.sum(episode_returns))
        all_returns.append(ep_return)
        all_lengths.append(step)
        if len(equity_curve) > 1:
            all_equity.extend(np.diff(equity_curve) / (np.array(equity_curve[:-1]) + 1e-12))
        if ep_return > 0:
            wins += 1

    ret_arr = np.array(all_returns)
    eq_arr = np.array(all_equity) if all_equity else np.zeros(1)

    return {
        "mean_return": float(np.mean(ret_arr)),
        "std_return": float(np.std(ret_arr, ddof=1)) if len(ret_arr) > 1 else 0.0,
        "sharpe": _compute_sharpe(eq_arr),
        "max_drawdown": _compute_max_drawdown(np.cumprod(1 + eq_arr)),
        "win_rate": wins / max(n_episodes, 1),
        "mean_episode_length": float(np.mean(all_lengths)),
    }


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------


class TradingMetricsCallback(BaseCallback):
    """Custom SB3 callback that logs portfolio metrics during training."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: list = []
        self._episode_lengths: list = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.verbose >= 1:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "portfolio_value" in info:
                    logger.debug(
                        "NAV=%.2f | DD=%.3f | Trades=%d",
                        info.get("portfolio_value", 0),
                        info.get("current_drawdown", 0),
                        info.get("trade_count", 0),
                    )


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


class PPOTradingAgent:
    """
    Proximal Policy Optimisation agent wrapping stable-baselines3 PPO.

    Designed for the TradingEnvironment with MultiInputPolicy (Dict obs space).
    """

    def __init__(
        self,
        env: gym.Env,
        policy: str = "MultiInputPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_env: bool = True,
        tensorboard_log: Optional[str] = None,
        seed: int = 42,
        verbose: int = 1,
        device: str = "auto",
    ) -> None:
        self.env = env
        self.normalize_env = normalize_env
        self.seed = seed
        self.verbose = verbose
        self._model: Optional[PPO] = None
        self._vec_env: Optional[DummyVecEnv] = None
        self._vec_normalize: Optional[VecNormalize] = None

        # Wrap environment
        monitored_env = Monitor(env)
        self._vec_env = DummyVecEnv([lambda: monitored_env])

        if normalize_env:
            self._vec_normalize = VecNormalize(
                self._vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )
            training_env = self._vec_normalize
        else:
            training_env = self._vec_env

        self._model = PPO(
            policy=policy,
            env=training_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )

        logger.info(
            "PPOTradingAgent initialised | lr=%.2e | n_steps=%d | batch=%d",
            learning_rate, n_steps, batch_size,
        )

    # ---------------------------------------------------------------- train --

    def train(
        self,
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: str = "./checkpoints/ppo",
        log_dir: str = "./logs/ppo",
        reset_num_timesteps: bool = True,
    ) -> "PPOTradingAgent":
        """
        Train the PPO agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to train for.
        eval_freq : int
            Evaluate every N steps.
        n_eval_episodes : int
            Episodes per evaluation.
        eval_env : gym.Env or None
            Held-out evaluation environment.  If None, uses the training env.
        checkpoint_dir : str
            Directory for model checkpoints.
        log_dir : str
            Directory for best-model saves (EvalCallback).
        reset_num_timesteps : bool
            Whether to reset the timestep counter.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        callbacks = []

        # Checkpoint callback – saves every eval_freq steps
        ckpt_cb = CheckpointCallback(
            save_freq=eval_freq,
            save_path=checkpoint_dir,
            name_prefix="ppo_trading",
            save_replay_buffer=False,
            save_vecnormalize=self.normalize_env,
            verbose=self.verbose,
        )
        callbacks.append(ckpt_cb)

        # Evaluation callback
        if eval_env is not None:
            monitored_eval = Monitor(eval_env)
            eval_vec = DummyVecEnv([lambda: monitored_eval])
            if self.normalize_env and self._vec_normalize is not None:
                eval_vec = VecNormalize(
                    eval_vec,
                    training=False,
                    norm_obs=True,
                    norm_reward=False,
                )
                eval_vec.obs_rms = self._vec_normalize.obs_rms
                eval_vec.ret_rms = self._vec_normalize.ret_rms

            eval_cb = EvalCallback(
                eval_env=eval_vec,
                best_model_save_path=log_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose,
            )
            callbacks.append(eval_cb)

        # Custom metrics callback
        callbacks.append(TradingMetricsCallback(verbose=self.verbose))

        callback_list = CallbackList(callbacks)

        logger.info(
            "Starting PPO training for %d timesteps ...", total_timesteps
        )
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )
        logger.info("PPO training complete.")
        return self

    # --------------------------------------------------------------- predict --

    def predict(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate an action from the current policy.

        Parameters
        ----------
        obs : dict or np.ndarray
            Observation from the environment.
        deterministic : bool
            If True, returns the modal action (no stochastic sampling).

        Returns
        -------
        action : np.ndarray
        state : np.ndarray or None
        """
        if self._model is None:
            raise RuntimeError("Model not initialised. Call train() first.")
        return self._model.predict(obs, deterministic=deterministic)

    # --------------------------------------------------------------- evaluate -

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate the agent over n_episodes.

        Returns
        -------
        dict with keys: mean_return, std_return, sharpe, max_drawdown,
                        win_rate, mean_episode_length
        """
        if self._model is None:
            raise RuntimeError("Model not initialised.")
        return _evaluate_agent(self._model, env, n_episodes)

    # ------------------------------------------------------------------ save --

    def save(self, path: str) -> None:
        """Save model and (optionally) VecNormalize statistics."""
        if self._model is None:
            raise RuntimeError("Nothing to save – model not trained.")
        self._model.save(path)
        if self._vec_normalize is not None:
            stats_path = str(path) + "_vecnorm.pkl"
            self._vec_normalize.save(stats_path)
            logger.info("VecNormalize stats saved to %s", stats_path)
        logger.info("PPO model saved to %s", path)

    # ------------------------------------------------------------------ load --

    @classmethod
    def load(
        cls,
        path: str,
        env: gym.Env,
        normalize_env: bool = True,
        vecnorm_path: Optional[str] = None,
        device: str = "auto",
    ) -> "PPOTradingAgent":
        """Load a previously saved PPOTradingAgent."""
        agent = cls.__new__(cls)
        agent.env = env
        agent.normalize_env = normalize_env
        agent.verbose = 1
        agent._vec_normalize = None

        monitored_env = Monitor(env)
        agent._vec_env = DummyVecEnv([lambda: monitored_env])

        if normalize_env:
            if vecnorm_path and Path(vecnorm_path).exists():
                agent._vec_normalize = VecNormalize.load(
                    vecnorm_path, agent._vec_env
                )
                agent._vec_normalize.training = False
                agent._vec_normalize.norm_reward = False
                training_env = agent._vec_normalize
            else:
                agent._vec_normalize = VecNormalize(
                    agent._vec_env, training=False
                )
                training_env = agent._vec_normalize
        else:
            training_env = agent._vec_env

        agent._model = PPO.load(path, env=training_env, device=device)
        logger.info("PPO model loaded from %s", path)
        return agent


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------


class SACTradingAgent:
    """
    Soft Actor-Critic agent wrapping stable-baselines3 SAC.

    SAC is an off-policy algorithm well-suited to continuous action spaces.
    Uses an experience replay buffer sized to hold millions of transitions.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: str = "MultiInputPolicy",
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: Union[str, float] = "auto",
        learning_starts: int = 10_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        action_noise_std: float = 0.1,
        tensorboard_log: Optional[str] = None,
        seed: int = 42,
        verbose: int = 1,
        device: str = "auto",
    ) -> None:
        self.env = env
        self.seed = seed
        self.verbose = verbose
        self._model: Optional[SAC] = None

        # Optional Gaussian exploration noise for SAC (usually handled via entropy)
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions),
        )

        # Wrap in Monitor + VecEnv
        monitored_env = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitored_env])

        self._model = SAC(
            policy=policy,
            env=vec_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            policy_kwargs=dict(
                net_arch=[256, 256],
                log_std_init=-3,
                use_sde=False,
            ),
        )

        logger.info(
            "SACTradingAgent initialised | lr=%.2e | buffer=%d | batch=%d",
            learning_rate, buffer_size, batch_size,
        )

    # ---------------------------------------------------------------- train --

    def train(
        self,
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: str = "./checkpoints/sac",
        log_dir: str = "./logs/sac",
        reset_num_timesteps: bool = True,
    ) -> "SACTradingAgent":
        """
        Train the SAC agent.

        Parameters
        ----------
        total_timesteps : int
            Total environment interaction steps.
        eval_freq : int
            How often (in steps) to run evaluation.
        n_eval_episodes : int
            Episodes per evaluation round.
        eval_env : gym.Env or None
            Separate evaluation environment.
        checkpoint_dir, log_dir : str
            Directories for checkpoints and best-model saves.
        reset_num_timesteps : bool
            Whether to reset internal step counter.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        callbacks: list = []

        ckpt_cb = CheckpointCallback(
            save_freq=eval_freq,
            save_path=checkpoint_dir,
            name_prefix="sac_trading",
            save_replay_buffer=True,
            verbose=self.verbose,
        )
        callbacks.append(ckpt_cb)

        if eval_env is not None:
            eval_vec = DummyVecEnv([lambda: Monitor(eval_env)])
            eval_cb = EvalCallback(
                eval_env=eval_vec,
                best_model_save_path=log_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose,
            )
            callbacks.append(eval_cb)

        callbacks.append(TradingMetricsCallback(verbose=self.verbose))
        callback_list = CallbackList(callbacks)

        logger.info("Starting SAC training for %d timesteps ...", total_timesteps)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )
        logger.info("SAC training complete.")
        return self

    # --------------------------------------------------------------- predict --

    def predict(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate an action.  SAC defaults to deterministic=True for live trading.
        """
        if self._model is None:
            raise RuntimeError("Model not initialised.")
        return self._model.predict(obs, deterministic=deterministic)

    # --------------------------------------------------------------- evaluate -

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate over n_episodes.

        Returns
        -------
        dict: mean_return, std_return, sharpe, max_drawdown, win_rate,
              mean_episode_length
        """
        if self._model is None:
            raise RuntimeError("Model not initialised.")
        return _evaluate_agent(self._model, env, n_episodes)

    # ------------------------------------------------------------------ save --

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save – model not trained.")
        self._model.save(path)
        # Also save replay buffer alongside
        buf_path = str(path) + "_replay_buffer"
        try:
            self._model.save_replay_buffer(buf_path)
            logger.info("Replay buffer saved to %s", buf_path)
        except Exception as exc:
            logger.warning("Could not save replay buffer: %s", exc)
        logger.info("SAC model saved to %s", path)

    # ------------------------------------------------------------------ load --

    @classmethod
    def load(
        cls,
        path: str,
        env: gym.Env,
        replay_buffer_path: Optional[str] = None,
        device: str = "auto",
    ) -> "SACTradingAgent":
        """Load a previously saved SACTradingAgent."""
        agent = cls.__new__(cls)
        agent.env = env
        agent.verbose = 1

        monitored_env = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitored_env])

        agent._model = SAC.load(path, env=vec_env, device=device)

        if replay_buffer_path and Path(replay_buffer_path).exists():
            try:
                agent._model.load_replay_buffer(replay_buffer_path)
                logger.info("Replay buffer loaded from %s", replay_buffer_path)
            except Exception as exc:
                logger.warning("Could not load replay buffer: %s", exc)

        logger.info("SAC model loaded from %s", path)
        return agent

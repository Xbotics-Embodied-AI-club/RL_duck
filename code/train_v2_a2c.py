"""第二版：A2C（带基线的策略梯度）。

相比 v1 多了一个 critic：用它估计的状态价值做基线，并用 GAE 算优势，
显著降低梯度方差，学得比 v1 稳、比 v1 快。
但相比 v3 仍然“朴素”：
  - 一段数据只用一遍（单 epoch、单 minibatch），不做重要性采样裁剪；
  - 学习率固定，不按 KL 自适应。
所以面对 小鸭子这种较难的任务，它会比完整 PPO 更不稳、上限更低。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import lightning as L
import torch
import wandb
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import NUM_ENVS, DuckEnv, task_slug  # noqa: E402
from model import ActorCritic, compute_gae  # noqa: E402


def default_checkpoint_root(run_name: str) -> Path:
    """给出这次训练存权重的目录。

    权重是大文件，按仓库约定落在共享数据根下、不进 git，所以路径从环境变量拼出来。

    Args:
        run_name: 本次训练的名字，同时用作目录名。

    Returns:
        存放 checkpoint 的目录路径。
    """
    root = os.environ.get("RL_DUCK_CKPT_ROOT")
    if root:
        return Path(root) / run_name
    return Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "rl_duck" / run_name


def save_checkpoint(path, model, optimizer, iteration, training_settings):
    """存一份可续训、也可直接拿去评测的权重。

    优化器状态和这次训练的全部设置一起存下来，评测脚本才能凭 checkpoint 自己重建
    出维度一致的网络，不必再猜环境配置。

    Args:
        path: 目标文件路径。
        model: 要保存的 `ActorCritic`。
        optimizer: 当前优化器。
        iteration: 当前是第几次迭代。
        training_settings: 本次训练的全部设置，一并写进文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "actor_critic": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_settings": training_settings,
        },
        path,
    )


class DuckRolloutDataset(IterableDataset):
    """采一段轨迹，用 critic 基线 + GAE 算优势，整段只产出一个 batch（用一遍）。"""

    def __init__(self, env, model, num_steps_per_env, gamma, lam):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield self.sample_rollout()

    def sample_rollout(self):
        """用当前策略采一段轨迹，并用 critic 估值 + GAE 算出优势。

        与 v1 唯一的区别就在这里：权重从「回报减常数均值」换成了「回报减 critic 估值」
        再做 GAE 平滑。

        Returns:
            含 obs / critic_obs / actions / returns / advantages 与本轮平均奖励的字典。
        """
        env = self.env
        num_envs = env.num_envs
        obs, critic_obs = env.get_observations()
        device = env.device
        obs_steps = torch.zeros(self.num_steps_per_env, num_envs, obs.shape[1], device=device)
        critic_obs_steps = torch.zeros(self.num_steps_per_env, num_envs, critic_obs.shape[1], device=device)
        actions_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        value_steps = torch.zeros(self.num_steps_per_env, num_envs, 1, device=device)
        reward_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        done_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        reward_sum = 0.0

        for step in range(self.num_steps_per_env):
            with torch.no_grad():
                actions, _log_probs, values, _means, _stds = self.model.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
            if "time_outs" in info:
                rewards = rewards + self.gamma * values.squeeze(-1) * info["time_outs"].float()

            obs_steps[step].copy_(obs)
            critic_obs_steps[step].copy_(critic_obs)
            actions_steps[step].copy_(actions)
            value_steps[step].copy_(values)
            reward_steps[step].copy_(rewards)
            done_steps[step].copy_(dones.float())

            self.model.update_actor_normalizer(next_obs)
            self.model.update_critic_normalizer(next_critic_obs)
            obs, critic_obs = next_obs, next_critic_obs
            reward_sum += float(rewards.mean().detach().cpu())

        with torch.no_grad():
            next_value = self.model.value(critic_obs)
        advantages, returns = compute_gae(
            reward_steps, done_steps, value_steps, next_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        batch_size = self.num_steps_per_env * num_envs
        return {
            "obs": obs_steps.reshape(batch_size, -1),
            "critic_obs": critic_obs_steps.reshape(batch_size, -1),
            "actions": actions_steps.reshape(batch_size, -1),
            "returns": returns.reshape(batch_size, 1),
            "advantages": advantages.reshape(batch_size),
            "reward_mean": torch.tensor(reward_sum / self.num_steps_per_env, device=device),
        }


class DuckData(L.LightningDataModule):
    """把持久环境交给 Trainer 的 LightningDataModule。

    环境要跨迭代活着（仿真状态连续推进），所以由它长期持有，而不是每轮新建。
    """
    def __init__(self, env, model, num_steps_per_env, gamma, lam):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam

    def train_dataloader(self):
        """每个 epoch 重建一次数据集。

        重建就意味着重新采样——on-policy 要求数据来自当前策略，这一条由 Trainer 的
        `reload_dataloaders_every_n_epochs=1` 和这里配合实现。

        Returns:
            包着在线数据集的 DataLoader；`batch_size=None` 表示数据集自己吐整批。
        """
        dataset = DuckRolloutDataset(self.env, self.model, self.num_steps_per_env, self.gamma, self.lam)
        return DataLoader(dataset, batch_size=None)


class DuckLightningA2C(L.LightningModule):
    """A2C：策略梯度用 GAE 优势，critic 回归 returns；不裁剪、不复用数据、固定 lr。"""

    def __init__(self, model, run_name, max_iterations, save_interval, checkpoint_dir,
                 training_settings, wandb_project, wandb_mode):
        super().__init__()
        self.model = model
        self.run_name = run_name
        self.max_iterations = max_iterations
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.training_settings = training_settings
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.latest_checkpoint = self.checkpoint_dir / "model_0.pt"
        self.wandb_run = None
        self.optimizer = None

        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.learning_rate = 1.0e-3

    def setup(self, stage):
        """训练开始前建好权重目录并起一个 W&B run。

        Args:
            stage: Lightning 传入的阶段名，只在 "fit" 阶段做事。
        """
        if stage != "fit":
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=self.wandb_project, name=self.run_name, mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(), config={**self.training_settings, "algo": "v2_a2c"},
        )

    def configure_optimizers(self):
        """把 actor、critic 和动作标准差一起交给优化器。

        与 v1 的区别：critic 这次要真训了。

        Returns:
            Adam 优化器。
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        """算一个 batch 的 A2C loss：策略梯度项 + critic 回归项 + 熵奖励。

        Args:
            batch: 采样端产出的一整段数据。
            batch_idx: Lightning 传入的批序号，本实现用不到。

        Returns:
            本步的 loss。
        """
        log_probs, entropy, values, _means = self.model.evaluate(
            batch["obs"], batch["critic_obs"], batch["actions"]
        )
        # 带基线的策略梯度：advantage 已减去 critic 估值；不做 ratio 裁剪。
        policy_loss = -(log_probs * batch["advantages"]).mean()
        value_loss = torch.square(values - batch["returns"]).mean()
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        iteration = self.current_epoch + 1
        metrics = {
            "reward": float(batch["reward_mean"].detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False)
        wandb.log(metrics, step=iteration)
        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            save_checkpoint(self.latest_checkpoint, self.model, self.optimizer, iteration, self.training_settings)
        return loss

    def teardown(self, stage):
        """训练结束后收尾，把 W&B run 正常关掉。

        Args:
            stage: Lightning 传入的阶段名。
        """
        if self.wandb_run is not None:
            self.wandb_run.finish()


def run_training(run_name, num_envs, max_iterations, num_steps_per_env, save_interval, device,
                 seed=1, checkpoint_dir=None, wandb_project="rl_duck", wandb_mode="offline"):
    """起一次完整训练：建环境、建模型、交给 Trainer 跑完。

    Args:
        run_name: 本次训练的名字，用作权重目录名与 W&B run 名。
        num_envs: 并行环境数。
        max_iterations: 训练迭代次数，一次迭代 = 采一段 rollout + 更新。
        num_steps_per_env: 每个环境每轮采多少步。
        save_interval: 每多少次迭代存一次权重。
        device: 仿真与训练所在设备。
        seed: 随机种子。
        checkpoint_dir: 权重目录，给 None 时按 run_name 自动拼。
        wandb_project: W&B 项目名。
        wandb_mode: W&B 模式，离线跑改成 offline 即可。

    Returns:
        最后一次存盘的 checkpoint 路径。
    """
    checkpoint_dir = checkpoint_dir or default_checkpoint_root(run_name)
    gamma, lam = 0.99, 0.95

    torch.manual_seed(seed)
    env = DuckEnv(num_envs=num_envs, device=device, seed=seed)
    env.reset()
    policy = ActorCritic(obs_dim=env.obs_dim, critic_obs_dim=env.critic_obs_dim, action_dim=env.action_dim)
    policy.to(env.device)

    training_settings = {
        "run_name": run_name, "num_envs": num_envs, "max_iterations": max_iterations,
        "num_steps_per_env": num_steps_per_env, "save_interval": save_interval, "device": device,
        "seed": seed, "checkpoint_dir": str(checkpoint_dir), "wandb_project": wandb_project,
        "wandb_mode": wandb_mode, "gamma": gamma, "lam": lam,
        "obs_dim": env.obs_dim, "critic_obs_dim": env.critic_obs_dim, "action_dim": env.action_dim,
    }

    data = DuckData(env, policy, num_steps_per_env, gamma, lam)
    model = DuckLightningA2C(policy, run_name, max_iterations, save_interval, checkpoint_dir,
                               training_settings, wandb_project, wandb_mode)
    trainer = L.Trainer(
        accelerator="gpu" if device != "cpu" and torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_iterations,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(model, data)
    return model.latest_checkpoint


def main():
    """按课堂预算跑一次 A2C 训练。"""
    run_training(
        run_name=f"{task_slug()}-a2c",
        num_envs=NUM_ENVS,
        max_iterations=3000,
        num_steps_per_env=24,
        save_interval=200,
        device="cuda:0",
        wandb_project="rl_duck",
        wandb_mode="offline",
    )


if __name__ == "__main__":
    main()

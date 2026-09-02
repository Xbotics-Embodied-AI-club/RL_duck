"""第三版：完整 PPO。

相比 v2 多了三件让训练更稳的关键机制：
  1. 重要性采样比值的裁剪（clip），允许同一段数据反复用多轮而不跑偏；
  2. 一段 rollout 切成多个 minibatch、训多个 epoch，样本效率更高；
  3. 按 KL 散度自适应调整学习率，更新步长自动收放。
这就是 BeyondMimic 用的那套核心，直接套到 小鸭子任务上。

再加一件只有**部分任务**要的：左右对称的镜像一致性损失（见 `mirror_loss`）。
奖励表、课程表、域随机化都在环境侧、由 mjlab 内部驱动，训练器免费继承；
对称性增广是唯一一件在**算法侧**的，环境不会替我们做，必须写在这里。
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import lightning as L
import torch
import wandb
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import MAX_ITERATIONS, NUM_ENVS, DuckEnv, task_slug, task_symmetry_cfg
from model import ActorCritic, compute_gae


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


def adapt_learning_rate_to_kl(optimizer, kl, desired_kl, min_lr=1.0e-5, max_lr=1.0e-2):
    """按新旧策略的 KL 距离自动收放学习率。

    clip 只让「走太远」无利可图，并不真的锁住步长；再加这一层，KL 超标就减速、
    偏小就加速，把每次更新的幅度稳在一条窄带里。

    Args:
        optimizer: 要调整的优化器。
        kl: 本次更新前后策略的 KL 距离。
        desired_kl: 希望维持的 KL 水平。
        min_lr: 学习率下限。
        max_lr: 学习率上限。

    Returns:
        调整后的学习率。
    """
    current_lr = optimizer.param_groups[0]["lr"]
    kl_value = float(kl.detach().cpu())
    if kl_value > 2.0 * desired_kl:
        current_lr = max(min_lr, current_lr / 1.5)
    elif 0.0 < kl_value < 0.5 * desired_kl:
        current_lr = min(max_lr, current_lr * 1.5)
    optimizer.param_groups[0]["lr"] = current_lr
    return current_lr


def resolve_mirror_function(symmetry_cfg):
    """把任务声明的镜像函数从"点分路径"变成可调用的函数。

    路径是任务配置里的一个字符串（`"包.模块.函数"`），不是我们这边的一个 import ——
    照抄成 import 就等于在这里复制了一份声明，上游换函数那天会静默用旧的那个。
    不用镜像损失的任务返回 None。

    Args:
        symmetry_cfg: 任务声明的 symmetry 配置字典，可以是 None。

    Returns:
        (镜像函数, 损失系数)；不启用时是 (None, 0.0)。

    Raises:
        RuntimeError: 配置说要用镜像损失，但没给镜像函数的路径。
    """
    if not symmetry_cfg or not symmetry_cfg.get("use_mirror_loss"):
        return None, 0.0
    dotted = symmetry_cfg.get("data_augmentation_func")
    if not dotted:
        raise RuntimeError(
            "任务声明了 use_mirror_loss，但没给 data_augmentation_func —— "
            "镜像函数只能从任务的声明里取，本训练器不替它猜一个。"
        )
    module_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attr), float(
        symmetry_cfg.get("mirror_loss_coeff", 0.5)
    )


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


def append_metrics(path, iteration, metrics):
    """把这一迭代的指标追加成一行 JSON。

    为什么不只依赖 W&B：离线 run 的指标写在一个二进制 `.wandb` 文件里，
    那是内部格式、没有稳定的读接口（实测拿 wandb 的内部 datastore 直接解会
    DecodeError）。而训练曲线是讲义的交付物之一，不能押在一个读不出来的文件上。
    一行 JSON 的代价可以忽略，换来的是「不装 wandb、不联网也能出图」。

    Args:
        path: 目标 `.jsonl` 文件路径。
        iteration: 当前迭代号。
        metrics: 本次迭代的指标字典。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"iteration": iteration, **metrics}, sort_keys=True) + "\n")


class DuckRolloutDataset(IterableDataset):
    """在线数据集：每轮先采一段轨迹，再切成若干 minibatch 逐个交给训练循环。"""

    def __init__(self, env, model, num_steps_per_env, gamma, lam, num_learning_epochs, num_mini_batches):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rollout = self.sample_rollout()
        yield from self.iter_mini_batches(rollout)

    def sample_rollout(self):
        """用当前策略采一段轨迹，并记下旧策略的对数概率与分布参数。

        采样时就把这些量存下来，是因为训练阶段要拿它们算概率比值和 KL；等到更新时
        策略已经变了，补不回来。

        Returns:
            含整段 rollout 与 GAE 优势的字典。
        """
        env = self.env
        num_envs = env.num_envs
        obs, critic_obs = env.get_observations()
        device = env.device
        obs_steps = torch.zeros(self.num_steps_per_env, num_envs, obs.shape[1], device=device)
        critic_obs_steps = torch.zeros(self.num_steps_per_env, num_envs, critic_obs.shape[1], device=device)
        actions_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        log_prob_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        value_steps = torch.zeros(self.num_steps_per_env, num_envs, 1, device=device)
        reward_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        done_steps = torch.zeros(self.num_steps_per_env, num_envs, device=device)
        action_mean_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        action_std_steps = torch.zeros(self.num_steps_per_env, num_envs, self.model.action_dim, device=device)
        reward_sum = 0.0

        for step in range(self.num_steps_per_env):
            with torch.no_grad():
                actions, log_probs, values, action_means, action_stds = self.model.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
            if "time_outs" in info:
                rewards = rewards + self.gamma * values.squeeze(-1) * info["time_outs"].float()

            obs_steps[step].copy_(obs)
            critic_obs_steps[step].copy_(critic_obs)
            actions_steps[step].copy_(actions)
            log_prob_steps[step].copy_(log_probs)
            value_steps[step].copy_(values)
            reward_steps[step].copy_(rewards)
            done_steps[step].copy_(dones.float())
            action_mean_steps[step].copy_(action_means)
            action_std_steps[step].copy_(action_stds)

            self.model.update_actor_normalizer(next_obs)
            self.model.update_critic_normalizer(next_critic_obs)
            obs, critic_obs = next_obs, next_critic_obs
            reward_sum += float(rewards.mean().detach().cpu())

        with torch.no_grad():
            next_value = self.model.value(critic_obs)
        advantages, returns = compute_gae(reward_steps, done_steps, value_steps, next_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        return {
            "obs": obs_steps,
            "critic_obs": critic_obs_steps,
            "actions": actions_steps,
            "log_probs": log_prob_steps,
            "values": value_steps,
            "returns": returns,
            "advantages": advantages,
            "action_means": action_mean_steps,
            "action_stds": action_std_steps,
            "reward_mean": torch.tensor(reward_sum / self.num_steps_per_env, device=device),
        }

    def iter_mini_batches(self, rollout):
        """把一整段 rollout 打散成若干 minibatch，重复吐几轮。

        PPO 的数据复用就实现在这里：Lightning 那边一行不用改，多轮复用完全由数据集完成。

        Args:
            rollout: `sample_rollout` 的产出。

        Yields:
            一个个 minibatch 字典。
        """
        batch_size = rollout["actions"].shape[0] * rollout["actions"].shape[1]
        mini_batch_size = batch_size // self.num_mini_batches
        usable_size = mini_batch_size * self.num_mini_batches
        reward_mean = rollout["reward_mean"]
        flat = {
            "obs": rollout["obs"].reshape(batch_size, -1),
            "critic_obs": rollout["critic_obs"].reshape(batch_size, -1),
            "actions": rollout["actions"].reshape(batch_size, -1),
            "log_probs": rollout["log_probs"].reshape(batch_size),
            "values": rollout["values"].reshape(batch_size, 1),
            "returns": rollout["returns"].reshape(batch_size, 1),
            "advantages": rollout["advantages"].reshape(batch_size),
            "action_means": rollout["action_means"].reshape(batch_size, -1),
            "action_stds": rollout["action_stds"].reshape(batch_size, -1),
        }
        # 同一段 rollout 重复多遍：这就是 PPO 的多轮 minibatch 更新。
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(usable_size, device=flat["actions"].device)
            for start in range(0, usable_size, mini_batch_size):
                selected = indices[start : start + mini_batch_size]
                mini_batch = {key: value[selected] for key, value in flat.items()}
                mini_batch["reward_mean"] = reward_mean
                yield mini_batch


class DuckData(L.LightningDataModule):
    """持有持久环境，每轮把用最新策略采到的新数据集交给 Trainer。"""

    def __init__(self, env, model, num_steps_per_env, gamma, lam, num_learning_epochs, num_mini_batches):
        super().__init__()
        self.env = env
        self.model = model
        self.num_steps_per_env = num_steps_per_env
        self.gamma = gamma
        self.lam = lam
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

    def train_dataloader(self):
        """每个 epoch 重建一次数据集。

        重建就意味着重新采样——on-policy 要求数据来自当前策略，这一条由 Trainer 的
        `reload_dataloaders_every_n_epochs=1` 和这里配合实现。

        Returns:
            包着在线数据集的 DataLoader；`batch_size=None` 表示数据集自己吐整批。
        """
        dataset = DuckRolloutDataset(
            env=self.env,
            model=self.model,
            num_steps_per_env=self.num_steps_per_env,
            gamma=self.gamma,
            lam=self.lam,
            num_learning_epochs=self.num_learning_epochs,
            num_mini_batches=self.num_mini_batches,
        )
        return DataLoader(dataset, batch_size=None)


class DuckLightningPPO(L.LightningModule):
    """只负责一个 minibatch 的 PPO loss、记录与保存；更新循环交给 Lightning。"""

    def __init__(self, model, run_name, max_iterations, save_interval, checkpoint_dir,
                 training_settings, wandb_project, wandb_mode, symmetry_cfg=None):
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
        self.latest_kl = torch.zeros(())
        self.epoch_records: list[dict[str, float]] = []

        self.value_loss_coef = 1.0
        self.use_clipped_value_loss = True
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.learning_rate = 1.0e-3
        self.desired_kl = 0.01
        self.mirror_function, self.mirror_loss_coeff = resolve_mirror_function(symmetry_cfg)

    def setup(self, stage):
        """训练开始前建好权重目录并起一个 W&B run。

        Args:
            stage: Lightning 传入的阶段名，只在 "fit" 阶段做事。
        """
        if stage != "fit":
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=self.run_name,
            mode=self.wandb_mode,
            dir=self.checkpoint_dir.as_posix(),
            config={**self.training_settings, "algo": "v3_ppo"},
        )

    def configure_optimizers(self):
        """把全部参数交给一个 Adam。

        Returns:
            Adam 优化器。
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def on_train_epoch_start(self):
        """每轮开始时清空本轮的指标缓存。"""
        self.epoch_records = []

    def training_step(self, mini_batch, batch_idx):
        """算一个 minibatch 的 PPO loss，并把指标攒进本轮缓存。

        Args:
            mini_batch: 数据集吐出的一个 minibatch。
            batch_idx: Lightning 传入的批序号，本实现用不到。

        Returns:
            本步的 loss。
        """
        loss, policy_loss, value_loss, entropy_loss, kl, mirror = self.loss_for_batch(mini_batch)
        self.latest_kl = kl.detach()
        record = {
            "reward": float(mini_batch["reward_mean"].detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "entropy": float(entropy_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "mirror_loss": float(mirror.detach().cpu()),
        }
        self.epoch_records.append(record)
        self.log_dict(record, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """每次真正迈步之前，先按 KL 把学习率调好。

        Args:
            optimizer: Lightning 传入的优化器。
        """
        adapt_learning_rate_to_kl(self.optimizer, self.latest_kl, self.desired_kl)

    def on_train_epoch_end(self):
        """把本轮各 minibatch 的指标平均后上报，并按需存盘。"""
        iteration = self.current_epoch + 1
        metrics = {key: sum(r[key] for r in self.epoch_records) / len(self.epoch_records) for key in self.epoch_records[0]}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        wandb.log(metrics, step=iteration)
        append_metrics(self.checkpoint_dir / "metrics.jsonl", iteration, metrics)
        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            save_checkpoint(self.latest_checkpoint, self.model, self.optimizer, iteration, self.training_settings)

    def mirror_loss(self, obs, critic_obs, action_means):
        """算左右镜像的一致性损失：把镜子里的观测喂进去，应当得到镜子里的动作。

        为什么它必须写在算法侧：这不是一条奖励，而是加在 loss 上的一项约束
        —— 「策略函数本身应当与左右镜像可交换」。环境不知道网络长什么样，替不了这件事。

        用它的理由（上游前滚翻那档的实测经验）：前滚翻是严格左右对称的动作，而策略很容易
        学成往一侧塌 —— 那是一条能量更低的斜路，正着翻要经过完全倒立的姿态。镜像一致性
        把"左偏"和"右偏"这两个解绑在一起，斜路的优势就消失了。左右不对称的任务
        （比如单脚踢球）绝不能开，它会直接把正确答案罚掉。

        镜像的排列与符号表由任务那一侧提供（61 维观测哪一位和哪一位换、哪几位要变号），
        我们这边只负责调用与算差 —— 那张表属于机器人的关节布局，不属于训练器。

        Args:
            obs: 一批 actor 观测。
            critic_obs: 同一批的 critic 观测；镜像函数按 TensorDict 收两组，故一并传入。
            action_means: 当前策略在 `obs` 上给出的动作均值。

        Returns:
            标量的镜像一致性损失；未启用镜像损失时是一个 0。
        """
        if self.mirror_function is None:
            return torch.zeros((), device=obs.device)
        batch_size = obs.shape[0]
        augmented, _ = self.mirror_function(
            None,
            TensorDict(
                {"actor": obs, "critic": critic_obs},
                batch_size=[batch_size],
                device=obs.device,
            ),
            None,
        )
        _, mirrored_actions = self.mirror_function(None, None, action_means)
        # 镜像函数把结果拼成 [原始; 镜像] 两半，我们只要后一半。
        mirrored_obs = augmented["actor"][batch_size:]
        target = mirrored_actions[batch_size:].detach()
        predicted = self.model.action_distribution(mirrored_obs).mean
        return torch.square(predicted - target).mean()

    def loss_for_batch(self, batch):
        """算一个 minibatch 上的各项 loss 与 KL。

        KL 在 `no_grad` 下算：它只用来调学习率，不参与反向传播。

        Args:
            batch: 一个 minibatch。

        Returns:
            (总 loss, 策略 loss, 价值 loss, 熵, KL, 镜像 loss)。
        """
        log_probs, entropy, values, action_means = self.model.evaluate(
            batch["obs"], batch["critic_obs"], batch["actions"]
        )
        with torch.no_grad():
            action_stds = self.model.std.clamp_min(1.0e-6).expand_as(action_means)
            old_stds = batch["action_stds"].clamp_min(1.0e-6)
            kl = torch.sum(
                torch.log(action_stds / old_stds + 1.0e-5)
                + (torch.square(old_stds) + torch.square(batch["action_means"] - action_means))
                / (2.0 * torch.square(action_stds))
                - 0.5,
                dim=-1,
            ).mean()

        ratio = torch.exp(log_probs - batch["log_probs"])
        unclipped = ratio * batch["advantages"]
        clipped = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                   * batch["advantages"])
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = self.value_loss(values, batch["values"], batch["returns"])
        entropy_loss = entropy.mean()
        mirror = self.mirror_loss(batch["obs"], batch["critic_obs"], action_means)
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy_loss
            + self.mirror_loss_coeff * mirror
        )
        return loss, policy_loss, value_loss, entropy_loss, kl, mirror

    def value_loss(self, values, old_values, returns):
        """critic 的回归损失，可选对新估值做对称裁剪。

        裁剪的思路与策略侧的 clip 一脉相承：不让 critic 的估值一次跳得离旧估值太远。

        Args:
            values: 当前 critic 的估值。
            old_values: 采样时记下的旧估值。
            returns: 回归目标。

        Returns:
            标量损失。
        """
        if not self.use_clipped_value_loss:
            return torch.square(values - returns).mean()
        value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
        return torch.max(torch.square(values - returns), torch.square(value_clipped - returns)).mean()

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
    num_learning_epochs, num_mini_batches = 5, 4

    torch.manual_seed(seed)
    env = DuckEnv(num_envs=num_envs, device=device, seed=seed)
    env.reset()
    policy = ActorCritic(obs_dim=env.obs_dim, critic_obs_dim=env.critic_obs_dim, action_dim=env.action_dim)
    policy.to(env.device)

    # 对称性只有任务自己知道要不要（前滚翻要，其余十几个都不要），所以问注册表。
    symmetry_cfg = task_symmetry_cfg(env.task)

    training_settings = {
        "run_name": run_name, "num_envs": num_envs, "max_iterations": max_iterations,
        "num_steps_per_env": num_steps_per_env, "save_interval": save_interval, "device": device,
        "seed": seed, "checkpoint_dir": str(checkpoint_dir), "wandb_project": wandb_project,
        "wandb_mode": wandb_mode, "gamma": gamma, "lam": lam,
        "num_learning_epochs": num_learning_epochs, "num_mini_batches": num_mini_batches,
        "obs_dim": env.obs_dim, "critic_obs_dim": env.critic_obs_dim, "action_dim": env.action_dim,
        "task": env.task, "symmetry": symmetry_cfg,
    }

    data = DuckData(env, policy, num_steps_per_env, gamma, lam, num_learning_epochs, num_mini_batches)
    model = DuckLightningPPO(policy, run_name, max_iterations, save_interval, checkpoint_dir,
                               training_settings, wandb_project, wandb_mode, symmetry_cfg)
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
    """按课堂预算跑一次完整 PPO 训练。"""
    run_training(
        run_name=f"{task_slug()}-ppo",
        num_envs=NUM_ENVS,
        max_iterations=MAX_ITERATIONS,
        num_steps_per_env=24,
        save_interval=200,
        device="cuda:0",
        wandb_project="rl_duck",
        wandb_mode="offline",
    )


if __name__ == "__main__":
    main()

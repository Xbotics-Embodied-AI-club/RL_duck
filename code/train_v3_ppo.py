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
from env import MAX_ITERATIONS, NUM_ENVS, SEED, TASK, DuckEnv, task_slug, task_symmetry_cfg
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


def check_mirror_contract(mirror_function, obs_dim, action_dim) -> None:
    """开训前验一次镜像函数的两条契约，不满足就停下。

    为什么值得花这几行：镜像损失依赖一张 61 维观测的置换与符号表（哪一位和哪一位换、
    哪几位变号），表由任务那一侧提供。**表接错了不会让训练崩** —— 它照样收敛，
    只是收敛到一个错的对称约束上，于是前滚翻继续往一侧塌，而那正是当初开镜像损失
    要修的现象。到那时没人分得清是"镜像损失没用"还是"镜像表接错了"。

    两条契约：
      ① 镜像是对合的：镜两次回到原值。表里有一位的符号或配对写错，这条就破。
      ② 返回值的前一半是原始输入。`mirror_loss` 靠切后一半拿镜像结果，
         这条同时把那个切片约定钉住 —— 上游哪天改成 [镜像; 原始] 会当场报错，
         而不是安静地把损失算成 0。

    Args:
        mirror_function: 已解析出来的镜像函数。
        obs_dim: actor 观测维度。
        action_dim: 动作维度。

    Raises:
        RuntimeError: 任一条契约不成立。
    """
    probe = torch.randn(4, obs_dim)
    critic_probe = torch.randn(4, obs_dim)
    once, _ = mirror_function(
        None, TensorDict({"actor": probe, "critic": critic_probe}, batch_size=[4]), None
    )
    if not torch.allclose(once["actor"][:4], probe, atol=1e-6):
        raise RuntimeError(
            "镜像函数返回值的前一半不等于输入 —— [原始; 镜像] 这个拼接约定变了。"
            "mirror_loss 靠切后一半取镜像结果，约定一变损失就算错，而训练照样跑。"
        )
    mirrored = once["actor"][4:]
    twice, _ = mirror_function(
        None, TensorDict({"actor": mirrored, "critic": critic_probe}, batch_size=[4]), None
    )
    if not torch.allclose(twice["actor"][4:], probe, atol=1e-6):
        raise RuntimeError(
            "镜像两次没有回到原值 —— 61 维观测的置换或符号表有位写错了。"
            "这种错不会让训练崩，只会让它收敛到一个错的对称约束上。"
        )
    actions = torch.randn(4, action_dim)
    _, act_once = mirror_function(None, None, actions)
    _, act_twice = mirror_function(None, None, act_once[4:])
    if not torch.allclose(act_once[:4], actions, atol=1e-6) or not torch.allclose(
        act_twice[4:], actions, atol=1e-6
    ):
        raise RuntimeError("动作侧的镜像表不满足「前一半是原始」或「镜两次回到原值」。")


def resolve_mirror_function(symmetry_cfg, obs_dim=None, action_dim=None):
    """把任务声明的镜像函数从"点分路径"变成可调用的函数，并当场验一次它的契约。

    路径是任务配置里的一个字符串（`"包.模块.函数"`），不是我们这边的一个 import ——
    照抄成 import 就等于在这里复制了一份声明，上游换函数那天会静默用旧的那个。
    不用镜像损失的任务返回 None。

    上游那个函数的第一个形参是 `env`，而它的 docstring 明写"未使用，只为接口兼容"，
    所以调用处一律传 `None`；这是照着上游的声明来的，不是漏了。

    Args:
        symmetry_cfg: 任务声明的 symmetry 配置字典，可以是 None。
        obs_dim: actor 观测维度，给出时会跑一次契约自检。
        action_dim: 动作维度，同上。

    Returns:
        (镜像函数, 损失系数)；不启用时是 (None, 0.0)。

    Raises:
        RuntimeError: 配置说要用镜像损失，但没给镜像函数的路径或没给系数；
            或者镜像函数的契约自检不通过。
    """
    # 上游那份配置里有**两个**独立开关，本训练器只实现了后者：
    #   · use_data_augmentation —— 把镜像样本追加进 batch（数据侧，等于把批量翻倍）
    #   · use_mirror_loss       —— 在 loss 上加一项镜像一致性（算法侧，我们实现的是这个）
    # 33 个任务里目前全都是 use_data_augmentation=False（前滚翻实测也是 False），
    # 所以"不实现"暂时没有代价。但**不能默默忽略它**：哪天上游把它打开，
    # 我们会只做一半的配方，而两个开关都在同一个字典里、日志上看不出差别。
    # ⇒ 请求了就报错，把"没实现"变成一件响的事。
    if symmetry_cfg and symmetry_cfg.get("use_data_augmentation"):
        raise RuntimeError(
            f"任务 {TASK} 声明了 use_data_augmentation=True，而本训练器只实现了 "
            "use_mirror_loss（loss 上的镜像一致性项），没有实现数据侧的镜像增广。"
            "照原样跑等于只执行了一半的配方 —— 要么实现它，要么明确决定不用它。"
        )
    if not symmetry_cfg or not symmetry_cfg.get("use_mirror_loss"):
        return None, 0.0
    dotted = symmetry_cfg.get("data_augmentation_func")
    if not dotted:
        raise RuntimeError(
            "任务声明了 use_mirror_loss，但没给 data_augmentation_func —— "
            "镜像函数只能从任务的声明里取，本训练器不替它猜一个。"
        )
    # 系数同样不给兜底值：凭空写一个 0.5 会在上游改键名那天静默改掉约束的强度。
    if "mirror_loss_coeff" not in symmetry_cfg:
        raise RuntimeError(
            "任务声明了 use_mirror_loss，但没给 mirror_loss_coeff —— "
            "约束强度必须由声明处给出，本训练器不替它定一个。"
        )
    module_name, _, attr = dotted.rpartition(".")
    mirror_function = getattr(importlib.import_module(module_name), attr)
    if obs_dim is not None and action_dim is not None:
        check_mirror_contract(mirror_function, obs_dim, action_dim)
    return mirror_function, float(symmetry_cfg["mirror_loss_coeff"])


def save_checkpoint(path, model, optimizer, iteration, training_settings):
    """存一份可续训、也可直接拿去评测的权重。

    优化器状态与 `training_settings` 一起存下来，评测脚本才能凭 checkpoint 自己重建
    出维度一致的网络、并把课程表对齐到同一档，不必再猜环境配置。

    **`training_settings` 不是"全部设置"**（先前这里就是这么写的，不准）：它有
    run 名、并行环境数、迭代数、每轮步数、存盘间隔、设备、种子、目录、W&B 设置、
    折扣与 GAE 系数、三个维度、任务 id（v3 另有 minibatch 划分与 symmetry 配置）——
    但**没有** learning_rate / clip_param / entropy_coef / value_loss_coef /
    desired_kl 这几个优化侧的超参。它们目前写死在各版本的 `__init__` 里，
    也就是说：**换了超参重跑，从 checkpoint 分不出这一份是哪套超参训的。**
    本篇三档对照用的是同一套超参，所以暂时没有代价；要做超参扫描时得先补上。

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
            # 日志累加的是**环境给的原始奖励**，不含下面那个超时自举项。
            # 三档同台图的纵轴靠这个数，而 v1 没有 critic、也就没有自举项 ——
            # 拿"加过自举"的数去和"没加"的比，纵轴就不是同一把尺子。
            # 实测口径：velocity-flat 的回合 1000 步、每轮采 24 步 ⇒ 约 0.1% 的
            # 环境步是超时步，那些步上加的是 γ·V(s)，V 的量级是 reward/(1−γ)，
            # 于是均值被抬高约一成；回合越短抬得越多（前滚翻 250 步一回合 ⇒ 数倍）。
            # 上游 rsl_rl 也是这么分的：自举加在 clone 上，日志用原始值。
            reward_sum += float(rewards.mean().detach().cpu())
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
        # 初值是 None，不是一个看起来像路径的假值：还没存过盘时调用方必须能分辨。
        # 先前这里是 `checkpoint_dir / "model_0.pt"`，那个文件从来不存在 ——
        # 一旦哪一轮没存上盘，run_training 就返回一条死路径而不报错。
        self.latest_checkpoint: Path | None = None
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
        self.latest_lr = self.learning_rate
        # 维度从模型自己身上取，不另传一遍 —— 契约自检要造和真实观测同形的探针。
        self.mirror_function, self.mirror_loss_coeff = resolve_mirror_function(
            symmetry_cfg,
            obs_dim=int(model.actor_mean.shape[0]),
            action_dim=int(model.action_dim),
        )

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
            batch_idx: Lightning 传入的批序号。签名是 Lightning 规定的，形参名不能改；
                本实现按顺序消费数据、不需要序号，所以显式 `del` 掉它 ——
                留着不用会被 lint 判为"接口没对齐"，而这里是真的不需要。

        Returns:
            本步的 loss。
        """
        del batch_idx
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

        调的是**Lightning 传进来的那个** optimizer，而不是 `self.optimizer`。
        今天两者是同一个对象（已核实：Lightning 的包装器 `LightningOptimizer.step` 把
        `self._optimizer` —— 原始的 Adam —— 传下去，钩子收到的就是它），所以两种写法
        都能跑。但改学习率必须改在"Lightning 真的会 step 的那个对象"上：一旦引入
        LR scheduler、梯度累积或精度插件，`self.optimizer` 就可能不再是被 step 的那个，
        于是 KL 自适应静默失效、训练照跑。

        Args:
            optimizer: Lightning 传入的优化器，就是本步会被 step 的那一个。
        """
        self.latest_lr = adapt_learning_rate_to_kl(optimizer, self.latest_kl, self.desired_kl)

    def on_train_epoch_end(self):
        """把本轮各 minibatch 的指标平均后上报，并按需存盘。"""
        iteration = self.current_epoch + 1
        # 存盘的判据只看迭代号，不受「这一轮有没有指标」影响。
        # 先前 `if not self.epoch_records: return` 挡在存盘前面：万一最后一轮没产出
        # 记录，最终 checkpoint 就不存了，而 run_training 返回一条死路径 ——
        # 一次三小时的训练「成功」结束、拿不到权重，且不报错。
        # v1/v2 没有这个问题：它们在 training_step 里存盘，不经过这个钩子。
        # 指标可以缺（那一轮就不 log），权重不能缺。
        if iteration % self.save_interval == 0 or iteration == self.max_iterations:
            self.latest_checkpoint = self.checkpoint_dir / f"model_{iteration}.pt"
            save_checkpoint(self.latest_checkpoint, self.model, self.optimizer, iteration,
                            self.training_settings)
        if not self.epoch_records:
            return
        metrics = {
            key: sum(r[key] for r in self.epoch_records) / len(self.epoch_records)
            for key in self.epoch_records[0]
        }
        # 学习率取上一次调整时的返回值，不再去 param_groups 里另读一遍 ——
        # 同一个事实两处各算一次，改了一处就会不一致。
        metrics["lr"] = self.latest_lr
        wandb.log(metrics, step=iteration)
        append_metrics(self.checkpoint_dir / "metrics.jsonl", iteration, metrics)

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
            stage: Lightning 传入的阶段名。签名由 Lightning 规定；收尾不分阶段，
                所以显式 `del` 掉，免得"未用形参"被读成接口没对齐。
        """
        del stage
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

    # 对称性只有任务自己知道要不要（33 个任务里只有前滚翻要），所以问注册表。
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
        seed=SEED,
        max_iterations=MAX_ITERATIONS,
        num_steps_per_env=24,
        # 存盘间隔可按进程覆盖。默认 200 是给完整训练用的；
        # 而"进化视频"要的是**最早那两百次迭代**里的样子 ——
        # 学会站、学会迈第一步全发生在那段，200 这个间隔一张都没留下。
        save_interval=int(os.environ.get("RL_DUCK_SAVE_INTERVAL") or 200),
        device="cuda:0",
        wandb_project="rl_duck",
        wandb_mode="offline",
    )


if __name__ == "__main__":
    main()

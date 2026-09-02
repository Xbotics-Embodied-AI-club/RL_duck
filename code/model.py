"""小鸭子三个版本共用的模型与优势估计工具。

`ActorCritic` 是一个普通的 PyTorch 模型（不含任何训练逻辑），训练生命周期交给各
`train_v*.py` 里的 LightningModule；`compute_gae` 是 v2/v3 用来把时序差分残差
平滑成优势的工具函数，v1 不用它。

讲义对应：第14讲 5.4 节（v1 走读）、6.3 节（GAE）、6.4 节（v2 走读）。
"""
from __future__ import annotations

import torch
from torch import nn


def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """用广义优势估计（GAE）把一段 rollout 换算成每步的优势与回报目标。

    从后往前递推，是因为第 t 步的优势要用到第 t+1 步的结果；正着走就得把整段
    存下来再回头算两遍。遇到回合结束（done）时把递推截断，免得把下一个回合的
    回报算进上一个回合。

    Args:
        rewards: 形状 (T, N) 的即时奖励，T 是采样步数，N 是并行环境数。
        dones: 形状 (T, N)，回合在该步结束则为 1。
        values: 形状 (T, N, 1)，critic 对每步状态的估值。
        next_value: 形状 (N, 1)，最后一步之后那个状态的估值，用来接上截断的尾巴。
        gamma: 折扣因子，控制"看多远"。
        lam: GAE 系数，在"偏差小但噪声大"与"噪声小但依赖 critic"之间取平衡。

    Returns:
        (advantages, returns) 两个张量。advantages 形状 (T, N)，直接当策略梯度
        的权重；returns 形状 (T, N, 1)，是 critic 要回归的目标。
    """
    rewards_3d = rewards.unsqueeze(-1) if rewards.ndim == 2 else rewards
    dones_3d = dones.unsqueeze(-1) if dones.ndim == 2 else dones
    values_3d = values.unsqueeze(-1) if values.ndim == 2 else values
    next_value_3d = next_value.unsqueeze(-1) if next_value.ndim == 1 else next_value
    advantages = torch.zeros_like(values_3d)
    last_advantage = torch.zeros_like(next_value_3d)
    for step in reversed(range(rewards_3d.shape[0])):
        next_values = next_value_3d if step == rewards_3d.shape[0] - 1 else values_3d[step + 1]
        not_done = 1.0 - dones_3d[step]
        delta = rewards_3d[step] + gamma * next_values * not_done - values_3d[step]
        last_advantage = delta + gamma * lam * not_done * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values_3d
    return advantages.squeeze(-1), returns


class ActorCritic(nn.Module):
    """三个版本共用的模型：actor 输出高斯动作分布，critic 估状态价值。

    带在线 running normalization（observation 标准化），维度由外部按环境实际
    观测维度传入，不在模型里写死。
    """

    def __init__(
        self,
        obs_dim,
        critic_obs_dim,
        action_dim,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        init_noise_std=1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.normalizer_epsilon = 1.0e-2
        self.register_buffer("actor_mean", torch.zeros(obs_dim))
        self.register_buffer("actor_var", torch.ones(obs_dim))
        self.register_buffer("actor_count", torch.tensor(0.0))
        self.register_buffer("critic_mean", torch.zeros(critic_obs_dim))
        self.register_buffer("critic_var", torch.ones(critic_obs_dim))
        self.register_buffer("critic_count", torch.tensor(0.0))
        self.actor = self.build_mlp(obs_dim, actor_hidden_dims, action_dim)
        self.critic = self.build_mlp(critic_obs_dim, critic_hidden_dims, 1)
        self.std = nn.Parameter(torch.full((action_dim,), init_noise_std))

    @staticmethod
    def build_mlp(input_dim, hidden_dims, output_dim):
        """堆一串「线性层 + ELU」，最后一层不带激活。

        actor 和 critic 用同一个构造器，只是输出维度不同——这样两边的容量完全
        对等，版本之间的差距才不会被网络结构的差异污染。

        Args:
            input_dim: 输入维度。
            hidden_dims: 各隐藏层宽度。
            output_dim: 输出维度，actor 是动作维数，critic 是 1。

        Returns:
            拼好的 `nn.Sequential`。
        """
        layers, last = [], input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(last, hidden), nn.ELU()]
            last = hidden
        layers.append(nn.Linear(last, output_dim))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def update_actor_normalizer(self, obs):
        """用新采到的一批观测更新 actor 侧的归一化统计量。

        强化学习的观测分布会随策略一起漂移，一次算好的均值方差很快就不适用，
        所以统计量要边采边更新。

        Args:
            obs: 形状 (N, obs_dim) 的一批 actor 观测。
        """
        self.actor_mean, self.actor_var, self.actor_count = self._update_stats(
            obs, self.actor_mean, self.actor_var, self.actor_count
        )

    @torch.no_grad()
    def update_critic_normalizer(self, obs):
        """同上，但更新的是 critic 侧的统计量。

        critic 看的观测比 actor 多（含特权信息），维度不同，所以两套统计量分开维护。

        Args:
            obs: 形状 (N, critic_obs_dim) 的一批 critic 观测。
        """
        self.critic_mean, self.critic_var, self.critic_count = self._update_stats(
            obs, self.critic_mean, self.critic_var, self.critic_count
        )

    def _update_stats(self, x, mean, var, count):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        new_count = count + batch_count
        rate = batch_count / new_count
        delta = batch_mean - mean
        new_mean = mean + rate * delta
        new_var = var + rate * (batch_var - var + delta * (batch_mean - new_mean))
        return new_mean, new_var, new_count

    def normalize_actor(self, obs):
        """把 actor 观测按当前统计量标准化。

        Args:
            obs: 原始 actor 观测。

        Returns:
            标准化后的观测，量级大致落在 1 附近。
        """
        return (obs - self.actor_mean) / (torch.sqrt(self.actor_var) + self.normalizer_epsilon)

    def normalize_critic(self, obs):
        """把 critic 观测按当前统计量标准化。

        Args:
            obs: 原始 critic 观测。

        Returns:
            标准化后的观测。
        """
        return (obs - self.critic_mean) / (torch.sqrt(self.critic_var) + self.normalizer_epsilon)

    def action_distribution(self, obs):
        """给出当前状态下的动作分布。

        均值由 actor 网络算出，标准差是一组与状态无关的可训练参数——探索强度
        由训练自己调，不随观测变化。

        Args:
            obs: 一批 actor 观测。

        Returns:
            逐关节独立的高斯分布 `torch.distributions.Normal`。
        """
        mean = self.actor(self.normalize_actor(obs))
        std = self.std.clamp_min(1.0e-6).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, critic_obs):
        """critic 给状态打分。

        Args:
            critic_obs: 一批 critic 观测（含特权信息）。

        Returns:
            形状 (N, 1) 的状态价值估计。
        """
        return self.critic(self.normalize_critic(critic_obs))

    def act(self, obs, critic_obs):
        """采样阶段用：采一个动作，并把训练要用的量一次性带回来。

        采样时就记下 log 概率与均值、标准差，是因为 PPO 训练时要拿"旧策略"的
        这几个量算概率比值和 KL；等到更新时策略已经变了，补不回来。

        Args:
            obs: actor 观测。
            critic_obs: critic 观测。

        Returns:
            (动作, 动作的对数概率, 状态价值, 分布均值, 分布标准差)。
        """
        dist = self.action_distribution(obs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, log_prob, self.value(critic_obs), dist.mean, dist.stddev

    def evaluate(self, obs, critic_obs, actions):
        """训练阶段用：拿当前策略重新评估一批已经采好的动作。

        Args:
            obs: actor 观测。
            critic_obs: critic 观测。
            actions: 采样阶段执行过的动作。

        Returns:
            (对数概率, 熵, 状态价值, 分布均值)。
        """
        dist = self.action_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(critic_obs), dist.mean

    def act_inference(self, obs):
        """部署 / 评测时用：直接取分布均值，不采样。

        评测要的是可复现的成绩，掺进随机探索就说不清分数是策略挣的还是运气挣的。

        Args:
            obs: actor 观测。

        Returns:
            确定性动作。
        """
        return self.actor(self.normalize_actor(obs))

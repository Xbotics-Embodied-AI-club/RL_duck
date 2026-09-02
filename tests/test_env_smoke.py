"""最小冒烟：环境构得起来、走得动一步、维度对得上。

这是本仓唯一的自动检查，挡的是最容易发生也最难察觉的那类事故 —— 依赖装歪、
task id 写错、上游改了观测布局。这些都不会立刻报错，而会在训练几小时之后
以一个莫名其妙的形状错误暴露出来。

跑法（在仓根）：
    uv run pytest tests/ -q
需要 CUDA：MuJoCo Warp 只跑在 GPU 上。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import env as duck_env

# 讲义第 2 节写死了这两个数字，正文里的每一处解释都依赖它们。
# 上游改了观测布局，这里要红。
EXPECTED_OBS_DIM = 61
EXPECTED_ACTION_DIM = 14

# 这一轮打算训的任务（走路 + 挑出来的几个动作）。少一个就说明 task id 写错了
# 或者上游把它改名了 —— 那种情况下 `TASK` 换一行的说法就不成立。
PLANNED_TASKS = (
    "Mjlab-Velocity-Flat-MicroDuck",
    "Mjlab-Velocity-Flat-MicroDuck-Rollers",
    "Mjlab-Spin-Flat-MicroDuck",
    "Mjlab-StandUp-Flat-MicroDuck",
    "Mjlab-BallKick-Flat-MicroDuck",
    "Mjlab-SitStand-Flat-MicroDuck",
    "Mjlab-GroundPick-Flat-MicroDuck",
    "Mjlab-Roulade-Flat-MicroDuck",
)

# 观测里第 6–19 位是 14 个关节的相对角。镜像表对这一段的置换与符号，和对**动作**的
# 置换与符号是同一份（`symmetry.py` 里 `_OBS_PERM[6:20] = 6 + _JOINT_PERM`），
# 所以"把这一段读出来当动作"这个策略是**严格左右等变**的 —— 下面用它当正对照。
_JOINT_POS_SLICE = slice(6, 20)
# 而这一段（角速度 3 + 重力 3 + 前 8 个关节）跨过了不同的置换块，不等变，当反对照。
_MIXED_SLICE = slice(0, 14)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp 需要 CUDA")


def test_planned_tasks_are_registered():
    """这一轮要训的每个 task id 都在注册表里。"""
    from mjlab.tasks.registry import list_tasks

    registered = set(list_tasks())
    missing = [t for t in PLANNED_TASKS if t not in registered]
    assert not missing, f"注册表里没有这些任务：{missing}"


def test_task_slug_is_a_pure_function_of_task():
    """目录名是从 task id 算出来的，同一个 task 永远得到同一个名字。"""
    assert duck_env.task_slug("Mjlab-Velocity-Flat-MicroDuck") == "velocity-flat"
    assert duck_env.task_slug("Mjlab-Velocity-Flat-MicroDuck-Rollers") == "velocity-flat-rollers"
    assert duck_env.task_slug("Mjlab-StandUp-Flat-MicroDuck") == "standup-flat"


def test_unknown_task_fails_loudly():
    """任务名写错要当场报错，不许悄悄回落到默认任务。"""
    with pytest.raises(ValueError, match="未注册的任务"):
        duck_env.DuckEnv(num_envs=2, task="Mjlab-No-Such-Task")


def test_symmetry_is_declared_by_the_task_not_by_us():
    """对称性开关只有前滚翻是开的，而且值是从 mjlab 注册表读的。

    这一条守的是"不在训练器里另抄一份任务名到开关的表"：如果哪天有人把开关硬编码
    过来，上游改了之后这里会红。
    """
    roulade = duck_env.task_symmetry_cfg("Mjlab-Roulade-Flat-MicroDuck")
    assert roulade is not None, "前滚翻应当声明了对称性增广"
    assert roulade["use_mirror_loss"] is True
    assert roulade["mirror_loss_coeff"] > 0
    assert "symmetry" in roulade["data_augmentation_func"]

    for task in ("Mjlab-StandUp-Flat-MicroDuck", "Mjlab-BallKick-Flat-MicroDuck"):
        assert duck_env.task_symmetry_cfg(task) is None, f"{task} 不该开对称性"


def test_episodic_tasks_get_the_whole_episode_in_keyframes():
    """每个任务被分到哪一类取帧窗口，逐个钉死。

    这一条抓的是一次真错：第一版阈值借用了绘图长度 260 步，而起身的回合是 300 步，
    于是它被判成"长回合"、照旧跳掉前 60 步 —— 而起身在第 15 步就已经站直。
    改完代码没看图，就会以为修好了。阈值改成按秒判之后，这里把分类逐任务锁住。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from render_gallery import EPISODIC_THRESHOLD_SECONDS, keyframe_window

    class _FakeEnv:
        def __init__(self, seconds):
            self.episode_seconds = seconds
            self.episode_steps = int(seconds * 50)

    # 走路：20 秒，周期性动作 ⇒ 跳掉启动瞬态。
    skip, steps = keyframe_window(_FakeEnv(20.0))
    assert skip > 0, "走路那种长回合应当跳掉启动瞬态"
    assert steps < 20.0 * 50, "走路不该把整个 20 秒都渲一遍"

    # episodic 那几个（起身 6 秒、坐站 6 秒、前滚翻 5 秒）⇒ 整段、从第 0 步。
    for seconds in (6.0, 5.0):
        skip, steps = keyframe_window(_FakeEnv(seconds))
        assert skip == 0, f"{seconds} 秒的回合应当从第 0 步看，实得 skip={skip}"
        assert steps == int(seconds * 50), f"{seconds} 秒的回合应当整段看，实得 {steps} 步"

    # 阈值本身要把这两簇分开，别贴着任一边。
    assert 6.0 < EPISODIC_THRESHOLD_SECONDS < 20.0


def test_lookat_without_world_camera_is_rejected():
    """`camera_lookat` 只在自由相机下生效，传错组合要当场报错。

    守的是一次真事故：出关键帧那处传了 lookat、没传 camera_origin，于是走 mjlab 默认的
    跟拍模式、lookat 成了死值 —— 而画面看着是对的（跟拍恰好把鸭子放中间），所以没人发现，
    注释还写着"实测比出来的"。前提不成立时必须有人喊，不能什么都不发生。
    """
    with pytest.raises(ValueError, match="camera_lookat 只在"):
        duck_env.DuckEnv(num_envs=2, camera_lookat=(0.0, 0.0, 0.18))
    with pytest.raises(ValueError, match="camera_lookat 只在"):
        duck_env.DuckEnv(num_envs=2, camera_lookat=(0.0, 0.0, 0.18), camera_origin="asset_root")


def test_curriculum_alignment_refuses_to_guess():
    """算不出课程表档位时要抛错，不许回落到第 0 档。

    第 0 档正是这个函数要修的错（评测会系统性地把任务测简单），所以"算不出来就按第 0 档"
    等于把已知错误的口径当默认值。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from rollout import align_curriculum

    with pytest.raises(RuntimeError, match="课程表对不齐"):
        align_curriculum(env=None, iteration=2000, settings={})
    with pytest.raises(RuntimeError, match="课程表对不齐"):
        align_curriculum(env=None, iteration=-1, settings={"num_steps_per_env": 24})


class _SliceActor(torch.nn.Module):
    """把观测的某一段直接当动作输出的假 actor，用来构造一个已知等变性的策略。"""

    def __init__(self, where: slice):
        super().__init__()
        self.where = where

    def forward(self, obs):
        """取出那一段。

        Args:
            obs: 一批观测。

        Returns:
            观测里 `self.where` 那一段。
        """
        return obs[:, self.where]


def _mirror_loss_of(actor_slice, tmp_path):
    """拿一个"读出观测某一段当动作"的策略，算它的镜像一致性损失。

    归一化统计量保持在初始值（均值 0、方差 1），此时 `normalize_actor` 只是乘一个
    标量，等变性不受影响 —— 这一点是本对照能成立的前提。

    Args:
        actor_slice: 要读出的观测段。
        tmp_path: pytest 给的临时目录，只用来满足构造函数。

    Returns:
        标量损失值。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from model import ActorCritic
    from train_v3_ppo import DuckLightningPPO

    torch.manual_seed(0)
    policy = ActorCritic(obs_dim=61, critic_obs_dim=61, action_dim=14)
    policy.actor = _SliceActor(actor_slice)
    module = DuckLightningPPO(
        model=policy, run_name="t", max_iterations=1, save_interval=1,
        checkpoint_dir=tmp_path, training_settings={}, wandb_project="t", wandb_mode="offline",
        symmetry_cfg=duck_env.task_symmetry_cfg("Mjlab-Roulade-Flat-MicroDuck"),
    )
    obs = torch.randn(8, 61)
    means = policy.action_distribution(obs).mean
    return float(module.mirror_loss(obs, obs, means))


def test_mirror_loss_vanishes_exactly_for_an_equivariant_policy(tmp_path):
    """严格左右等变的策略，镜像损失必须是 0；不等变的必须明显大于 0。

    这两条一起才有意义。只测"等变的是 0"抓不到"把拼接结果取错了一半"这类错
    —— 取错一半时算的是 ‖μ(o) − 镜像(μ(o))‖²，对随机观测它不是 0，会被反对照抓到。
    """
    equivariant = _mirror_loss_of(_JOINT_POS_SLICE, tmp_path)
    mixed = _mirror_loss_of(_MIXED_SLICE, tmp_path)
    assert equivariant < 1e-10, f"等变策略的镜像损失应当是 0，实得 {equivariant}"
    assert mixed > 0.1, f"不等变策略的镜像损失应当明显大于 0，实得 {mixed}"


@needs_cuda
def test_env_builds_and_steps():
    """构环境、走一步，观测 61 维、动作 14 维。"""
    e = duck_env.DuckEnv(num_envs=4, device="cuda:0", seed=0)
    try:
        assert e.obs_dim == EXPECTED_OBS_DIM, f"actor 观测维度变了：{e.obs_dim}"
        assert e.action_dim == EXPECTED_ACTION_DIM, f"动作维度变了：{e.action_dim}"
        assert e.critic_obs_dim >= e.obs_dim, "critic 应当看到不少于 actor 的信息"

        obs, critic_obs = e.reset()
        assert obs.shape == (4, EXPECTED_OBS_DIM)
        assert critic_obs.shape == (4, e.critic_obs_dim)

        actions = torch.zeros(4, EXPECTED_ACTION_DIM, device=e.device)
        obs, critic_obs, rewards, dones, info = e.step(actions)
        assert obs.shape == (4, EXPECTED_OBS_DIM)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert torch.isfinite(obs).all(), "观测里出现了非有限值"
        assert torch.isfinite(rewards).all(), "奖励里出现了非有限值"
        assert "time_outs" in info, "PPO 要用 time_outs 给超时回合补自举回报"
    finally:
        e.close()

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

import env as duck_env  # noqa: E402

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
)

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

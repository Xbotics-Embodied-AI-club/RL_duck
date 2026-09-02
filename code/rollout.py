"""按统一口径评测三个版本训练出的策略，并录成对照视频。

讲义里 5.6、6.5、7.7 节的每一个"评测平均奖励"和"摔倒率"都由本文件产出：同一个
环境、同样 16 个并行环境跑 400 步、一律取确定性动作。口径统一，版本之间的数字
才可比。摘要写成 `result/<task-slug>/<run>.json`，视频写成同名 `.mp4`。

讲义对应：1.5 节（评测口径）、5.5 节（关联代码）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import DuckEnv, task_slug  # noqa: E402
from model import ActorCritic  # noqa: E402


def result_root() -> Path:
    """给出本模块结果目录的路径。

    结果按任务分目录，目录名从 env.py 的 `TASK` 算出来——换任务不会覆盖上一个任务的结果。

    Returns:
        `result/<task-slug>/` 的绝对路径。
    """
    return Path(__file__).resolve().parents[1] / "result" / task_slug()


def load_policy(checkpoint, device):
    """从 checkpoint 里恢复一个可推理的策略。

    网络的三个维度不写死在这里，而是从 checkpoint 存的 `training_settings` 里读——
    环境一改维度就跟着变，硬编码迟早对不上。

    Args:
        checkpoint: checkpoint 文件路径。
        device: 模型放到哪个设备上。

    Returns:
        (模型, 该 checkpoint 对应的训练迭代数)。
    """
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = data.get("training_settings", {})
    model = ActorCritic(
        obs_dim=settings["obs_dim"],
        critic_obs_dim=settings["critic_obs_dim"],
        action_dim=settings["action_dim"],
    )
    model.load_state_dict(data["actor_critic"])
    model.eval()
    model.to(device)
    return model, int(data.get("iteration", -1))


def _recorded_frame(frame) -> np.ndarray:
    frame = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame


def run_rollout(checkpoint, run_name, num_envs=16, num_steps=400, device="cuda:0", seed=1):
    """跑一段确定性 rollout，落盘评测摘要与视频。

    动作取 `act_inference`（分布均值）而不是采样，评测成绩才不掺探索噪声。

    Args:
        checkpoint: 要评测的权重文件。
        run_name: 结果文件名前缀，同时写进摘要里便于溯源。
        num_envs: 并行环境数，三个版本必须一致。
        num_steps: 每个环境跑多少步。
        device: 仿真与推理所在设备。
        seed: 随机种子，固定它三个版本才面对同一批初始状态。

    Returns:
        评测摘要字典，字段与落盘的 `.json` 一致。
    """
    torch.manual_seed(seed)
    out_dir = result_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_output = out_dir / f"{run_name}.json"
    video_output = out_dir / f"{run_name}.mp4"

    env = DuckEnv(num_envs=num_envs, device=device, seed=seed, render_mode="rgb_array")
    rewards, action_abs_means, frames = [], [], []
    done_count = 0
    try:
        model, iteration = load_policy(checkpoint, env.device)
        obs, _critic_obs = env.reset()
        for _ in range(num_steps):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic_obs, reward, done, _info = env.step(actions)
            rewards.append(float(reward.mean().detach().cpu()))
            action_abs_means.append(float(actions.detach().abs().mean().cpu()))
            done_count += int(done.detach().sum().cpu())
            frame = env.render()
            if frame is not None:
                frames.append(_recorded_frame(frame))
    finally:
        env.close()

    summary = {
        "run_name": run_name,
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_iteration": iteration,
        "video_output": str(video_output),
        "video_frames": int(len(frames)),
        "num_envs": int(num_envs),
        "steps": int(num_steps),
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "action_abs_mean": float(sum(action_abs_means) / len(action_abs_means)) if action_abs_means else 0.0,
        "done_count": int(done_count),
        "done_fraction": float(done_count / (num_envs * num_steps)) if num_envs and num_steps else 0.0,
    }
    json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if frames:
        fps = float(getattr(env, "metadata", {}).get("render_fps", 50.0))
        media.write_video(str(video_output), frames, fps=fps)
    return summary


def main():
    """依次评测三个版本各自的最终 checkpoint。"""
    root = os.environ.get("RL_DUCK_CKPT_ROOT")
    trained = Path(root) if root else Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "rl_duck"

    # 对三个版本各跑一遍对照（用各自最终 checkpoint）。
    slug = task_slug()
    runs = {f"{slug}-{algo}": f"{slug}-{algo}/model_3000.pt" for algo in ("reinforce", "a2c", "ppo")}
    for run_name, rel in runs.items():
        checkpoint = trained / rel
        if not checkpoint.exists():
            print(f"skip {run_name}: {checkpoint} not found")
            continue
        summary = run_rollout(checkpoint=checkpoint, run_name=run_name, device="cuda:0")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

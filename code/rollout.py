"""按统一口径评测三个版本训练出的策略，并录成对照视频。

讲义里 5.6、6.5、7.7 节的每一个"评测平均奖励"和"摔倒率"都由本文件产出：同一个
环境、同样 16 个并行环境跑 400 步、一律取确定性动作。口径统一，版本之间的数字
才可比。摘要写成 `result/<task-slug>/<run>.json`，视频写成同名 `.mp4`。

**"摔倒率"这个词只对得上 `terminated_fraction`，不对 `done_fraction`。** 环境把
"摔倒终止"和"超时截断"合并成一个 `done` 返回（对回报截断的作用相同），而这里评测
要区分它们：走路是长回合、400 步内几乎不超时，把 `done` 当摔倒读恰好没错；但起身、
坐站、前滚翻的回合只有 250–300 步，400 步里必然超时重置，那时 `done` 里绝大部分是
"时间到了"。把它当摔倒率写进讲义，就是把一个纯由回合长度决定的数字当成策略质量。

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
from env import MAX_ITERATIONS, DuckEnv, task_slug
from model import ActorCritic


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
        (模型, 该 checkpoint 对应的训练迭代数, 那次训练的全部设置)。设置一并回传是因为
        评测还要用它把环境的课程表对齐到同一档（见 `align_curriculum`）。
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
    return model, int(data.get("iteration", -1)), settings


def align_curriculum(env, iteration, settings) -> int:
    """把环境的课程表对齐到这份权重训练到的那一档，**必须在 reset 之前调**。

    不对齐会让评测系统性地把任务测简单：课程表按环境步分段，新建环境从第 0 步开始，
    于是起身那档评测时"仰躺"这种初始姿态的概率是 0（它要到第 600 迭代才开始出现），
    各惩罚项也停在最轻的一档。数字照样出、图照样渲、脚本照样退 0。

    档位不用另存一个字段：每次迭代恰好推进 `num_steps_per_env` 个环境步，
    所以第 N 档权重对应 N × num_steps_per_env 步，从 checkpoint 里已有的两个字段就能算。

    Args:
        env: 已建好、**还没 reset** 的 `DuckEnv`。
        iteration: checkpoint 的迭代号。
        settings: checkpoint 里的 `training_settings`。

    Returns:
        对齐到的环境步数。

    Raises:
        RuntimeError: checkpoint 里缺迭代号或每轮步数，算不出档位。**不回落到第 0 档** ——
            那正是本函数要修的错，回落等于把已知错误的口径当成默认值继续跑，
            而那一行提示会被几百步的进度刷过去。三份训练脚本写 `training_settings`
            时这两个键必带，所以缺字段只可能是 checkpoint 不是本仓产的。
    """
    steps_per_iter = int(settings.get("num_steps_per_env", 0))
    if iteration <= 0 or steps_per_iter <= 0:
        raise RuntimeError(
            "课程表对不齐：checkpoint 里 "
            f"iteration={iteration!r}、training_settings['num_steps_per_env']={steps_per_iter!r}，"
            "至少有一个缺失或非正。评测必须知道这份权重训到了课程表的哪一档 —— "
            "按第 0 档跑会系统性地把任务测简单，而且不报错。"
        )
    step = iteration * steps_per_iter
    env.set_curriculum_step(step)
    print(f"课程表对齐到第 {step} 环境步（迭代 {iteration} × 每轮 {steps_per_iter} 步）。")
    return step


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

    # 分辨率与取景必须显式给。mjlab 的默认是 320×240、相机 5 米外近乎水平 ——
    # 那样录出来的评测视频里鸭子只有几十像素、上面还压着一条黑边，而且它不报错。
    # 这里与关键帧用同一套取景，两者才对得上。
    env = DuckEnv(
        num_envs=num_envs,
        device=device,
        seed=seed,
        render_mode="rgb_array",
        shadows=False,
        render_size=(1280, 960),
        camera_distance=0.9,
        camera_elevation=-15.0,
        # 跟拍：策略在按速度指令行走，固定机位下它几秒就出画，长镜头会变成一段空地板。
        camera_origin="asset_root",
    )
    rewards, action_abs_means, frames = [], [], []
    # 逐项奖励：mjlab 把每一项的回合累计放在 extras["log"] 里。总奖励会骗人 ——
    # 它可能全靠正则项在涨而目标动作从没发生，所以要看**主任务那一项**。
    # 这些任务都没有定义成功率指标（它们是连续奖励任务，不是成败任务），
    # 逐项奖励是我们能拿到的最接近「有没有做成那件事」的东西。
    term_sums: dict[str, float] = {}
    term_count = 0
    # 回合是怎么结束的，两种分开数。`terminated & ~time_outs` 才是"摔了/被判失败"；
    # 光看合并后的 done，短回合任务会把每次超时重置都记成一次摔倒。
    terminated_count = 0
    timeout_count = 0
    try:
        model, iteration, settings = load_policy(checkpoint, env.device)
        # 顺序是硬的：先对齐课程表，再 reset —— 初始姿态的混合是在 reset 那一刻按
        # 当前档位抽的，reset 之后再改就只能影响下一个回合。
        curriculum_step = align_curriculum(env, iteration, settings)
        obs, _critic_obs = env.reset()
        for _ in range(num_steps):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic_obs, reward, _done, info = env.step(actions)
            rewards.append(float(reward.mean().detach().cpu()))
            time_outs = info["time_outs"]
            terminated = info["terminated"]
            # 同一步里两个信号可能同时为真（撞上边界那一步刚好也超时），
            # 那一次回合结束只算一次，归给"终止"这一类。
            terminated_count += int(terminated.detach().sum().cpu())
            timeout_count += int((time_outs & ~terminated).detach().sum().cpu())
            log = info.get("mjlab_extras", {}).get("log", {})
            if log:
                term_count += 1
                for key, value in log.items():
                    term_sums[key] = term_sums.get(key, 0.0) + float(
                        value.mean() if hasattr(value, "mean") else value
                    )
            action_abs_means.append(float(actions.detach().abs().mean().cpu()))
            frame = env.render()
            if frame is not None:
                frames.append(_recorded_frame(frame))
    finally:
        env.close()

    summary = {
        "run_name": run_name,
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_iteration": iteration,
        "curriculum_step": curriculum_step,
        "video_output": str(video_output),
        "video_frames": len(frames),
        "num_envs": int(num_envs),
        "steps": int(num_steps),
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "action_abs_mean": float(sum(action_abs_means) / len(action_abs_means)) if action_abs_means else 0.0,
        # 分母是**回合数**，不是环境步数：说"率"的时候想的是"多少个回合是摔着结束的"。
        # 一个回合都没结束（走路 400 步不到一个回合长度）时两个率都给 0，
        # 而 episodes = 0 这个字段会把"没有样本"和"零摔倒"区分开。
        "episodes": int(terminated_count + timeout_count),
        "terminated_count": int(terminated_count),
        "timeout_count": int(timeout_count),
        "terminated_fraction": float(terminated_count / (terminated_count + timeout_count))
        if (terminated_count + timeout_count)
        else 0.0,
        "timeout_fraction": float(timeout_count / (terminated_count + timeout_count))
        if (terminated_count + timeout_count)
        else 0.0,
        "reward_terms": {k: v / term_count for k, v in sorted(term_sums.items())} if term_count else {},
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
    runs = {
        f"{slug}-{algo}": f"{slug}-{algo}/model_{MAX_ITERATIONS}.pt"
        for algo in ("reinforce", "a2c", "ppo")
    }
    for run_name, rel in runs.items():
        checkpoint = trained / rel
        if not checkpoint.exists():
            print(f"skip {run_name}: {checkpoint} not found")
            continue
        summary = run_rollout(checkpoint=checkpoint, run_name=run_name, device="cuda:0")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

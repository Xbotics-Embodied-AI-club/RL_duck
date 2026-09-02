"""把三级阶梯的训练曲线画成一张对照图。

数据从 W&B 的离线运行目录读（训练一律 `WANDB_MODE=offline`，事后 `wandb sync`），
所以不联网也能出图。横轴用**环境步数**而不是墙钟：墙钟依赖当时跑在哪张卡上、
有没有别的任务在抢，换台机器数字就变；环境步数是算法本身消耗的经验量，跨机器可比。

讲义对应：5.6 / 6.5 节（各自的曲线）、7.7 节（三个算法同台）。

跑法（在仓根，训完之后）：
    uv run python code/plot_reward_curves.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import NUM_ENVS, task_slug

# 三个版本在图上的固定顺序与颜色。顺序就是讲义里的演进顺序，别按字母排。
ALGOS = (
    ("reinforce", "v1 REINFORCE", "#9e9e9e"),
    ("a2c", "v2 A2C", "#1f77b4"),
    ("ppo", "v3 PPO", "#d62728"),
)
NUM_STEPS_PER_ENV = 24  # 与三份 train 脚本的 main() 一致；一次迭代 = NUM_ENVS × 这个数


def use_project_font():
    """把讲义指定的字体装进 matplotlib；取不到就报错停下。

    不回落系统字体是有原因的：回落之后图**照样渲出来**、脚本**照样退 0**，
    只是中文变成另一套字形，跟书里其它图不一致 —— 这种错要到排版时才看得见。

    Raises:
        RuntimeError: `XBOTICS_FIG_FONT` 没设，或者指向的文件不存在。
    """
    path = os.environ.get("XBOTICS_FIG_FONT", "")
    if not path or not Path(path).is_file():
        raise RuntimeError(
            "出图字体取不到：请把 XBOTICS_FIG_FONT 指向那份合并字体（西文 Times New Roman + 中文宋体）。"
            f"当前值 = {path!r}"
        )
    font_manager.fontManager.addfont(path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def read_offline_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """从一个 W&B 离线运行目录里读出「迭代 → 平均奖励」序列。

    离线 run 把每条 log 写成 `wandb-history.jsonl` 的一行，直接按行解析即可，
    不需要连 W&B 服务。

    Args:
        run_dir: 形如 `.../wandb/offline-run-<时间>-<id>/` 的目录。

    Returns:
        (迭代序号数组, 平均奖励数组)，按迭代升序。
    """
    history = run_dir / "files" / "wandb-history.jsonl"
    if not history.is_file():
        history = run_dir / "wandb-history.jsonl"
    steps, rewards = [], []
    with history.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "reward" not in row:
                continue
            steps.append(int(row.get("_step", len(steps) + 1)))
            rewards.append(float(row["reward"]))
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(rewards)[order]


def find_run_dir(wandb_root: Path, run_name: str) -> Path | None:
    """在离线 wandb 目录里找到某个 run 名对应的那个目录。

    同一个 run 名可能跑过多次（重跑、续训），取最后修改的那个 —— 也就是最新一次。

    Args:
        wandb_root: 存放 `offline-run-*` 的目录。
        run_name: 训练时给的 run 名。

    Returns:
        找到的目录，没找到返回 None。
    """
    candidates = []
    for d in sorted(wandb_root.glob("offline-run-*")):
        meta = d / "files" / "wandb-metadata.json"
        if not meta.is_file():
            continue
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("name") == run_name:
                candidates.append(d)
        except json.JSONDecodeError:
            continue
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    # 退一步：按目录名里带的 run 名匹配（metadata 缺失时）
    named = sorted(wandb_root.glob(f"offline-run-*{run_name}*"))
    return named[-1] if named else None


def smooth(values: np.ndarray, window: int = 21) -> np.ndarray:
    """滑动平均，把单点噪声压下去，趋势才看得出来。

    Args:
        values: 原始序列。
        window: 窗口长度，会被压到不超过序列长度的奇数。

    Returns:
        与输入等长的平滑序列。
    """
    if values.size < 3:
        return values
    window = min(window, values.size if values.size % 2 else values.size - 1)
    if window < 3:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def main():
    """出两张图：三个算法同台，以及各自单独一张。"""
    use_project_font()
    slug = task_slug()
    ckpt_root = Path(
        os.environ.get("RL_DUCK_CKPT_ROOT")
        or Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "rl_duck"
    )
    out_dir = Path(__file__).resolve().parents[1] / "result" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    for algo, label, color in ALGOS:
        run_name = f"{slug}-{algo}"
        wandb_root = ckpt_root / run_name / "wandb"
        if not wandb_root.is_dir():
            print(f"跳过 {run_name}：{wandb_root} 不存在（还没训 / 目录不对）")
            continue
        run_dir = find_run_dir(wandb_root, run_name)
        if run_dir is None:
            print(f"跳过 {run_name}：{wandb_root} 下没找到它的离线 run")
            continue
        iters, rewards = read_offline_run(run_dir)
        if iters.size == 0:
            print(f"跳过 {run_name}：离线 run 里没有 reward 记录")
            continue
        series[algo] = (iters * NUM_ENVS * NUM_STEPS_PER_ENV, rewards, label, color)
        print(f"{run_name}: {iters.size} 个记录点，最后 reward = {rewards[-1]:.4f}")

    if not series:
        print("一条曲线都没读到，先训练再来出图。")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for algo, _label, _color in ALGOS:
        if algo not in series:
            continue
        env_steps, rewards, label, color = series[algo]
        ax.plot(env_steps, rewards, color=color, alpha=0.22, linewidth=0.8)
        ax.plot(env_steps, smooth(rewards), color=color, linewidth=1.8, label=label)
    ax.set_xlabel("环境步数")
    ax.set_ylabel("每步平均奖励")
    ax.set_title("同一个任务、同一份预算，三个算法的学习曲线")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    combined = out_dir / "ladder-reward.png"
    fig.savefig(combined)
    plt.close(fig)
    print(f"写出 {combined}")

    for algo, (env_steps, rewards, label, color) in series.items():
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
        ax.plot(env_steps, rewards, color=color, alpha=0.22, linewidth=0.8)
        ax.plot(env_steps, smooth(rewards), color=color, linewidth=1.8)
        ax.set_xlabel("环境步数")
        ax.set_ylabel("每步平均奖励")
        ax.set_title(label)
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        single = out_dir / f"reward-{algo}.png"
        fig.savefig(single)
        plt.close(fig)
        print(f"写出 {single}")


if __name__ == "__main__":
    main()

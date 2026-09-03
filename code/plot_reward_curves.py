"""把三级阶梯的训练曲线画成一张对照图。

数据读每个 run 目录下的 `metrics.jsonl`（训练时逐迭代追加的那份）。**不读 W&B 的
离线目录**，理由要说准，因为这里有个容易过度概括的坑：

· **还没 `wandb sync` 的离线 run**，`files/` 里只有 `requirements.txt`，
  指标全在二进制 `run-*.wandb` 里；`wandb-history.jsonl` 这个文件名**从来不存在**
  （那是旧版 W&B 的格式）。本脚本先前正是照它去读，于是"一条曲线都没读到"
  却仍然退 0，看着像"你还没训练"。
· **`wandb sync` 之后**，同一个目录里会多出 `config.yaml` / `wandb-metadata.json` /
  `wandb-summary.json`。实测（velocity-flat-ppo 那档 sync 之后）`config.yaml` 里
  确实有 `num_envs: 4096` 与 `num_steps_per_env: 24` —— **它是一条可用的来源，
  而且不需要 torch。** 先前这段注释写"config.yaml 一个都不存在"，那句话只对
  没 sync 的 run 成立，写成通则就是过度概括。

仍然选 checkpoint 而不选 `config.yaml` 的理由：**sync 是收尾动作**，训练刚跑完、
还没 sync 的时候也要能出图；而 checkpoint 从第一次存盘起就在，且它正是评测那条
路径已经在读的同一个文件。⇒ 一份数据一个来源，不必为了省一次 torch.load 再引入
一条"要看是否 sync 过"的分支。

**判据不能建在一个不存在的文件上** —— 这条教训成立，只是原因不是"config.yaml 不存在"，
而是"`wandb-history.jsonl` 这个名字压根不属于当前版本的 W&B"。

横轴用**环境步数**而不是墙钟：墙钟依赖当时跑在哪张卡上、有没有别的任务在抢，
换台机器数字就变；环境步数是算法本身消耗的经验量，跨机器可比。换算所需的
`num_envs` / `num_steps_per_env` 从**该 run 自己的 checkpoint** 里读
（`save_checkpoint` 把 `training_settings` 一并存了进去），不用当前进程的环境变量 ——
批量跑时每档的 `RL_DUCK_NUM_ENVS` 可以不同，拿当前值去换算会整倍数地算错，
而且图照样画得出来、看不出错。

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
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import NUM_ENVS, result_base, task_slug

# 三个版本在图上的固定顺序与颜色。顺序就是讲义里的演进顺序，别按字母排。
ALGOS = (
    ("reinforce", "v1 REINFORCE", "#9e9e9e"),
    ("a2c", "v2 A2C", "#1f77b4"),
    ("ppo", "v3 PPO", "#d62728"),
)
NUM_STEPS_PER_ENV = 24  # 与三份 train 脚本的 main() 一致；读不到 run 自己的设置时的回落值


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
    family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False
# **数学符号用 STIX。** 两件事分开看：
    # ① `font.family` 管不到 `$...$` —— mathtext 默认走 DejaVu，图上的 $a_t$
    #    与周围的宋体/Times 正文不是一套字；
    # ② 把 mathtext 指到这份合并体（custom）能让字形对上，但那份字体没有斜体面，
    #    变量会渲成**直立**的，而 PDF 正文里 xelatex 把 $a_t$ 渲成斜体 —— 两边又不一致。
    # STIX 是按 Times 度量设计的数学字体，斜体、字重、笔画粗细都与 Times 正文匹配，
    # 是这类"中文正文 + Times 西文 + 行内公式"排版的标准解。
    plt.rcParams["mathtext.fontset"] = "stix"


def read_metrics(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """从 run 目录的 `metrics.jsonl` 里读出「迭代 → 平均奖励」序列。

    **只保留最后一次运行那一段。** 这个文件是追加写的，同一个 run 名重跑或续训时
    新的迭代号会从小数字重新开始，于是文件里躺着多段。若把整份按迭代号排序，
    两段会交错混在一起 —— 画出来是一条上下横跳的锯齿，看着像"训练不稳定"，
    其实是两次运行被叠在了一起。实测 velocity-flat-reinforce 那份就是
    3153 行旧段 + 新段，共 5164 行。判据：迭代号不再递增的地方就是重启点。

    Args:
        run_dir: 形如 `<ckpt_root>/<run_name>/` 的目录。

    Returns:
        (迭代序号数组, 平均奖励数组)，按迭代升序；文件不存在或没有 reward 时返回空数组。
    """
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return np.asarray([]), np.asarray([])
    steps, rewards = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "reward" not in row or "iteration" not in row:
                continue
            steps.append(int(row["iteration"]))
            rewards.append(float(row["reward"]))
    start = 0
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            start = i
    return np.asarray(steps[start:]), np.asarray(rewards[start:])


def read_run_settings(run_dir: Path) -> dict:
    """从该 run 自己的 checkpoint 里取出这次训练的设置。

    `save_checkpoint` 把整份 `training_settings` 存进了每个 `model_*.pt`，
    所以 run 目录是自描述的，不必再存一份侧写文件、也不必信当前进程的环境变量。
    `weights_only=False` 是必需的：这个字典里除了数字还有 symmetry 配置对象。

    Args:
        run_dir: 形如 `<ckpt_root>/<run_name>/` 的目录。

    Returns:
        `training_settings` 字典；没有 checkpoint 或里面没存设置时返回空字典。
    """
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return {}
    blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    return blob.get("training_settings") or {}


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
    """出两张图：三个算法同台，以及各自单独一张。

    Raises:
        SystemExit: 三档一条曲线都没读到 —— 这时候不能退 0，否则"图没出来"会被
            当成"训练还没跑"，而真正的原因（路径不对、文件名对不上）不会有人去查。
    """
    use_project_font()
    slug = task_slug()
    ckpt_root = Path(
        os.environ.get("RL_DUCK_CKPT_ROOT")
        or Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "rl_duck"
    )
    # 落点与其它出图脚本共用一处（`env.result_base()`，认 RL_DUCK_RESULT_ROOT）。
    # 先前这里写死了仓内 result/：给了环境变量之后，曲线写进仓内、
    # 而收件的 collect_delivery 去仓外找，于是报"三张图缺失"。
    out_dir = result_base() / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    for algo, label, color in ALGOS:
        run_name = f"{slug}-{algo}"
        run_dir = ckpt_root / run_name
        if not run_dir.is_dir():
            print(f"跳过 {run_name}：{run_dir} 不存在（还没训 / 目录不对）")
            continue
        iters, rewards = read_metrics(run_dir)
        if iters.size == 0:
            print(f"跳过 {run_name}：{run_dir / 'metrics.jsonl'} 里没有可用记录")
            continue
        settings = read_run_settings(run_dir)
        # 两个因子**分别**判来源。先前只看 num_envs 就打印 "checkpoint"，
        # 于是步数因子悄悄回落时诊断行仍然说"来自 checkpoint" —— 诊断行本身在说谎。
        n_envs = settings.get("num_envs") or NUM_ENVS
        n_steps = settings.get("num_steps_per_env") or NUM_STEPS_PER_ENV
        src_envs = "ckpt" if settings.get("num_envs") else f"回落{NUM_ENVS}"
        src_steps = "ckpt" if settings.get("num_steps_per_env") else f"回落{NUM_STEPS_PER_ENV}"
        if "回落" in src_envs or "回落" in src_steps:
            # 盘上真有 8192 与 4096 两种 run，横轴因子差两倍。回落只能是"没别的办法"，
            # 不能是默认路径 —— 所以它必须显眼。
            print(f"⚠ {run_name}: 换算因子有一项取不到该 run 自己的设置，"
                  f"横轴可能整倍数偏差（envs={src_envs}, steps={src_steps}）")
        series[algo] = (iters * n_envs * n_steps, rewards, label, color)
        print(
            f"{run_name}: 迭代 {iters[0]}–{iters[-1]}（{iters.size} 点），"
            f"每迭代 {n_envs}({src_envs})×{n_steps}({src_steps}) 环境步，"
            f"末段 reward = {rewards[-1]:.4f}"
        )

    if not series:
        raise SystemExit(f"三档一条曲线都没读到，检查 {ckpt_root} 下有没有 {slug}-* 的 run 目录")

    # 横轴要可比，前提是三档的「每迭代环境步数」一致；不一致就说出来，别让读者以为同预算。
    factors = {int(s[0][1] - s[0][0]) if s[0].size > 1 else 0 for s in series.values()}
    if len(factors) > 1:
        print(f"⚠ 三档的每迭代环境步数不一致：{sorted(factors)} —— 横轴仍可比，但"
              f"「同一份预算」这句话在讲义里要改。")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for algo, _label, _color in ALGOS:
        if algo not in series:
            continue
        env_steps, rewards, label, color = series[algo]
        ax.plot(env_steps, rewards, color=color, alpha=0.22, linewidth=0.8)
        ax.plot(env_steps, smooth(rewards), color=color, linewidth=1.8, label=label)
    ax.set_xlabel("环境步数")
    ax.set_ylabel("每步平均奖励")
    # 图上不写标题：图题是正文那一行的事，烧进 PNG 之后排版换了就动不了，
    # 而且与图题逐字重复。哪条线是哪个算法由图例说明。
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    combined = out_dir / "ladder-reward.png"
    fig.savefig(combined)
    plt.close(fig)
    print(f"写出 {combined}")

    for algo, (env_steps, rewards, _label, color) in series.items():
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
        ax.plot(env_steps, rewards, color=color, alpha=0.22, linewidth=0.8)
        ax.plot(env_steps, smooth(rewards), color=color, linewidth=1.8)
        ax.set_xlabel("环境步数")
        ax.set_ylabel("每步平均奖励")
        # 不烧标题：单档曲线是哪一档，正文的图题里写着。
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        single = out_dir / f"reward-{algo}.png"
        fig.savefig(single)
        plt.close(fig)
        print(f"写出 {single}")


if __name__ == "__main__":
    main()

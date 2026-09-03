"""把该交付的图和视频从过程产物里挑出来，拷进 `交付/`。

为什么要有这么一个脚本，而不是渲完手动挑：**交付件是什么，必须只有一处声明**。
过程目录里同一个动作有十几档权重的图、还有探针留下的残片，
靠记忆去挑，挑错一次没人看得出来 —— 讲义里那张"最终策略"可能摆的是第 400 迭代那版。
下面的 `FIGURES` 与 `VIDEOS` 就是那唯一一处声明：**源文件写死、交付名写死**，
源文件不在就报错停下，不静默跳过。

跑法（在仓根）：
    RL_DUCK_RESULT_ROOT=<过程产物根> uv run python code/collect_delivery.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from env import result_base
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DELIVERY = REPO / "交付"

# 讲义正文引用的图。左边是过程产物里的相对路径，右边是交付名。
# **正文里的图引用只认右边这一列** —— 换了权重档就改这里，正文一个字都不用动。
FIGURES: tuple[tuple[str, str], ...] = (
    ("cover.png", "00-开篇动作带.png"),
    ("velocity-flat/parallel-iter6000.png", "01-并行训练一屏.png"),
    ("velocity-flat/reward-reinforce.png", "02-v1-学习曲线.png"),
    ("velocity-flat/keyframes-v1-reinforce.png", "03-v1-一整个回合.png"),
    ("velocity-flat/reward-a2c.png", "04-v2-学习曲线.png"),
    ("velocity-flat/keyframes-v2-a2c.png", "05-v2-一整个回合.png"),
    ("velocity-flat/ladder-reward.png", "06-三档学习曲线.png"),
    ("velocity-flat/keyframes-iter6000.png", "07-v3-走路.png"),
    ("velocity-flat-rollers/parallel-iter2000.png", "08-轮滑.png"),
    ("spin-flat/parallel-iter2000.png", "09-旋转.png"),
    ("groundpick-flat/keyframes-iter2000.png", "10-取物.png"),
    ("roulade-flat/keyframes-iter6000.png", "11-前滚翻.png"),
)

# 交付视频。每个动作两版：单只看得清姿态，六只看得出这是同一套策略在一群身上跑。
VIDEOS: tuple[tuple[str, str], ...] = (
    ("velocity-flat/交付-走路-v3PPO.mp4", "单只-走路-v3PPO.mp4"),
    ("velocity-flat/交付-走路-v1REINFORCE.mp4", "单只-走路-v1REINFORCE.mp4"),
    ("velocity-flat/交付-走路-v2A2C.mp4", "单只-走路-v2A2C.mp4"),
    ("velocity-flat-rollers/交付-轮滑.mp4", "单只-轮滑.mp4"),
    ("spin-flat/交付-旋转.mp4", "单只-旋转.mp4"),
    ("groundpick-flat/交付-取物.mp4", "单只-取物.mp4"),
    ("roulade-flat/交付-前滚翻.mp4", "单只-前滚翻.mp4"),
    ("velocity-flat/parallel-iter6000.mp4", "六只-走路.mp4"),
    ("velocity-flat-rollers/parallel-iter2000.mp4", "六只-轮滑.mp4"),
    ("spin-flat/parallel-iter2000.mp4", "六只-旋转.mp4"),
    ("groundpick-flat/parallel-iter2000.mp4", "六只-取物.mp4"),
    ("roulade-flat/parallel-iter6000.mp4", "六只-前滚翻.mp4"),
    ("velocity-flat/evolution.mp4", "进化-走路-六档迭代.mp4"),
)


# 白边的判据与边距（bd xb-h26t）：非白像素包围盒，阈值 248、留 20 像素。
_WHITE_LEVEL = 248
_WHITE_PAD = 20


def crop_white_border(src: Path, dst: Path) -> None:
    """把图四周的白边裁掉再写到交付目录。

    为什么要裁：matplotlib 的画布按格子数算高，帧是 4:3 时每格上下都富余，
    出来的 PNG 纵向留白能占到 22%（实测 `keyframes-iter2000.png`）。
    排版上这条留白表现为"图与图注之间空一大条"，而**收 width% 治不了它** ——
    等比缩放留白跟着缩，相对位置一点不变。

    Args:
        src: 过程产物里的原图。
        dst: 交付目录里的目标路径。
    """
    img = Image.open(src).convert("RGB")
    mask = (np.asarray(img) < _WHITE_LEVEL).any(axis=2)
    ys, xs = np.where(mask)
    if ys.size == 0:
        shutil.copy2(src, dst)
        return
    h, w = mask.shape
    box = (max(int(xs.min()) - _WHITE_PAD, 0), max(int(ys.min()) - _WHITE_PAD, 0),
           min(int(xs.max()) + 1 + _WHITE_PAD, w), min(int(ys.max()) + 1 + _WHITE_PAD, h))
    img.crop(box).save(dst)


def copy_set(pairs: tuple[tuple[str, str], ...], out_dir: Path) -> list[str]:
    """按声明拷一组文件，返回缺失的源文件清单。

    Args:
        pairs: (过程产物相对路径, 交付名) 的序列。
        out_dir: 交付子目录。

    Returns:
        缺失的源文件相对路径（调用方决定怎么报）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for src_rel, dst_name in pairs:
        src = result_base() / src_rel
        if not src.is_file():
            missing.append(src_rel)
            continue
        if src.suffix.lower() == ".png":
            crop_white_border(src, out_dir / dst_name)
        else:
            shutil.copy2(src, out_dir / dst_name)
        print(f"  {src_rel}  ->  {dst_name}")
    return missing


def main() -> None:
    """把声明的图与视频拷进 `交付/`。

    Raises:
        SystemExit: 有声明了却不存在的源文件 —— 缺一件就停，不给一份不完整的交付。
    """
    print(f"过程产物根：{result_base()}")
    print("图：")
    missing = copy_set(FIGURES, DELIVERY / "图")
    print("视频：")
    missing += copy_set(VIDEOS, DELIVERY / "视频")
    if missing:
        print("★ 这些声明了的源文件不在，交付不完整：", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        raise SystemExit(1)
    print(f"齐了：{len(FIGURES)} 张图、{len(VIDEOS)} 段视频 -> {DELIVERY}")


if __name__ == "__main__":
    main()

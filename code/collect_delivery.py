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
from paths import result_base
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DELIVERY = REPO / "交付"
# 讲义正文引用的图落在 docs 里，与讲义同处 —— 交付目录只放视频与权重，
# 讲义与图不在那里再出现一份（一份东西两个位置，改一处忘一处只是时间问题）。
FIGURE_DIR = REPO / "docs/assets/figures"

# 讲义正文引用的图。左边是过程产物里的相对路径，右边是 `docs/assets/figures/` 下的名字。
# **正文里的图引用只认右边这一列** —— 换了权重档就改这里，正文一个字都不用动。
FIGURES: tuple[tuple[str, str], ...] = (
    ("cover.png", "00-开篇动作带.png"),
    ("velocity-flat/parallel-velocity-flat-ppo-iter6000.png", "01-并行训练一屏.png"),
    ("velocity-flat/reward-reinforce.png", "02-v1-学习曲线.png"),
    ("velocity-flat/keyframes-velocity-flat-reinforce-iter6000-16f4c.png", "03-v1-一整个回合.png"),
    ("velocity-flat/reward-a2c.png", "04-v2-学习曲线.png"),
    ("velocity-flat/keyframes-velocity-flat-a2c-iter6000-16f4c.png", "05-v2-一整个回合.png"),
    ("velocity-flat/ladder-reward.png", "06-三档学习曲线.png"),
    ("velocity-flat/keyframes-velocity-flat-ppo-iter6000-6f3c.png", "07-v3-走路.png"),
    ("velocity-flat-rollers/parallel-velocity-flat-rollers-ppo-iter2000.png", "08-轮滑.png"),
    ("spin-flat/parallel-spin-flat-ppo-iter2000.png", "09-旋转.png"),
    ("groundpick-flat/keyframes-groundpick-flat-ppo-iter2000-6f3c.png", "10-取物.png"),
    ("roulade-flat/keyframes-roulade-flat-ppo-iter6000-12f3c.png", "11-前滚翻.png"),
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
    ("velocity-flat/parallel-velocity-flat-ppo-iter6000.mp4", "六只-走路.mp4"),
    ("velocity-flat-rollers/parallel-velocity-flat-rollers-ppo-iter2000.mp4", "六只-轮滑.mp4"),
    ("spin-flat/parallel-spin-flat-ppo-iter2000.mp4", "六只-旋转.mp4"),
    ("groundpick-flat/parallel-groundpick-flat-ppo-iter2000.mp4", "六只-取物.mp4"),
    ("roulade-flat/parallel-roulade-flat-ppo-iter6000.mp4", "六只-前滚翻.mp4"),
    ("velocity-flat/evolution.mp4", "进化-走路-八档迭代.mp4"),
)


# 白边的判据与边距：非白像素包围盒，阈值 248、留 20 像素。
_WHITE_LEVEL = 248
_WHITE_PAD = 20

# 黑点的判据。渲染器会在画面上留下孤立的纯黑小方块 —— 实测是渲染侧留下的，不是编码
# 造成的（同一批帧直接存 PNG 与转成 mp4 再读回来，极暗像素数一模一样；crf 16 也一样），
# 所以重渲一遍治不了它，只能在收件这一步补掉。
#
# 判据三条同时成立才补，为的是**不碰真内容**：
#   ① 极暗（亮度 < 20）—— 那些方块是纯黑的；
#   ② 连通块小（≤ 400 像素）—— 机器人身上的深色件是大块连通区域；
#   ③ 四周一圈是中间调（中位亮度 25–160）—— 地板是这个区间。
# 第三条是关键的那道闸：曲线图上的黑字与坐标轴也满足①②，但它们四周是**近白**的画布，
# 被第三条挡在外面。改完逐张比过：三张曲线图零像素变化。
_SPECK_DARK = 20
_SPECK_MAX_AREA = 400
_SPECK_RING_LO, _SPECK_RING_HI = 25, 160


def crop_white_border(src: Path, dst: Path) -> None:
    """把图四周的白边裁掉再写到交付目录。

    为什么要裁：matplotlib 的画布按格子数算高，帧是 4:3 时每格上下都富余，
    出来的 PNG 纵向留白能占到 22%（实测 `keyframes-iter2000.png`）。
    排版上这条留白表现为"图与图注之间空一大条"，而**收 width% 治不了它** ——
    等比缩放留白跟着缩，相对位置一点不变。

    顺手把渲染器留下的孤立黑点补掉（见 `fill_render_specks`）—— 它们和白边一样
    是"渲出来就有、重渲还有"的东西，收件是唯一能一次治完所有图的位置。

    Args:
        src: 过程产物里的原图。
        dst: 交付目录里的目标路径。
    """
    img = Image.open(src).convert("RGB")
    img, specks = fill_render_specks(img)
    if specks:
        print(f"      补掉 {specks} 个渲染黑点")
    mask = (np.asarray(img) < _WHITE_LEVEL).any(axis=2)
    ys, xs = np.where(mask)
    if ys.size == 0:
        shutil.copy2(src, dst)
        return
    h, w = mask.shape
    box = (max(int(xs.min()) - _WHITE_PAD, 0), max(int(ys.min()) - _WHITE_PAD, 0),
           min(int(xs.max()) + 1 + _WHITE_PAD, w), min(int(ys.max()) + 1 + _WHITE_PAD, h))
    img.crop(box).save(dst)


def _dark_blobs(dark: np.ndarray, max_area: int) -> list[list[tuple[int, int]]]:
    """把极暗掩码里的连通块找出来，只留面积不超过 `max_area` 的那些。

    自己写的广度优先，不引 scipy：一张图上极暗像素才一千来个，遍历代价可以忽略，
    而少一个依赖意味着收件这一步在任何机器上都跑得动（它已经不依赖仿真器了）。

    Args:
        dark: (H, W) 的布尔掩码，True 表示极暗。
        max_area: 超过这个面积就不算黑点，直接丢弃（机器人身上的深色件是大块）。

    Returns:
        每个小连通块的像素坐标列表。
    """
    h, w = dark.shape
    seen = np.zeros_like(dark)
    blobs = []
    for y0, x0 in zip(*np.where(dark), strict=True):
        if seen[y0, x0]:
            continue
        stack, pix = [(int(y0), int(x0))], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            if len(pix) > max_area:
                break
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pix) <= max_area:
            blobs.append(pix)
    return blobs


def fill_render_specks(img: Image.Image) -> tuple[Image.Image, int]:
    """把渲染器留下的孤立纯黑小方块补成周围的颜色。

    补的值取那一圈的中位色，不做插值：那些方块落在地板上，地板是平缓渐变，
    中位色看不出接缝；而插值会在方块边缘留一道更显眼的糊边。

    Args:
        img: RGB 图。

    Returns:
        (补过的图, 补掉了几个块)。
    """
    a = np.asarray(img).copy()
    lum = a.mean(axis=2)
    blobs = _dark_blobs(lum < _SPECK_DARK, _SPECK_MAX_AREA)
    h, w = lum.shape
    filled = 0
    for pix in blobs:
        ys = [p[0] for p in pix]
        xs = [p[1] for p in pix]
        y0, y1 = max(min(ys) - 3, 0), min(max(ys) + 4, h)
        x0, x1 = max(min(xs) - 3, 0), min(max(xs) + 4, w)
        patch = lum[y0:y1, x0:x1]
        ring = patch[patch >= _SPECK_DARK]
        if ring.size == 0:
            continue
        med = float(np.median(ring))
        # 四周不是中间调就不动它 —— 近白的四周说明这是画布上的字，不是地板上的黑点。
        if not _SPECK_RING_LO <= med <= _SPECK_RING_HI:
            continue
        block = a[y0:y1, x0:x1]
        mask = lum[y0:y1, x0:x1] >= _SPECK_DARK
        if not mask.any():
            continue
        color = np.median(block[mask], axis=0)
        for y, x in pix:
            a[y, x] = color
        filled += 1
    return Image.fromarray(a), filled


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
    missing = copy_set(FIGURES, FIGURE_DIR)
    print("视频：")
    missing += copy_set(VIDEOS, DELIVERY / "视频")
    if missing:
        print("★ 这些声明了的源文件不在，交付不完整：", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        raise SystemExit(1)
    print(f"齐了：{len(FIGURES)} 张图 -> {FIGURE_DIR}，"
          f"{len(VIDEOS)} 段视频 -> {DELIVERY / '视频'}")


if __name__ == "__main__":
    main()

"""把同一个任务在不同训练程度下的表现接成一段"进化"视频，每一段左上角标着迭代数。

为什么值得单独有这么一段：讲义里那三条学习曲线说的是"分数在涨"，
而分数涨了到底长什么样，曲线是答不了的 —— 从原地抽搐、到踉跄两步、到稳稳迈步，
这个过程只有连着看才成立。一张张静态图放在一起也不行：走路是过程，定格拍不出过程。

标注是**烧进画面**的，这一处与本项目"图上不写标题"的规矩不冲突：
这段视频的全部意义就是"哪一段是第几次迭代"，标注不是装饰，是内容本身。

跑法（在仓根）：
    RL_DUCK_TASK=Mjlab-Velocity-Flat-MicroDuck \
    RL_DUCK_CKPT_ROOT=/mnt/nas_datasets/models/trained/rl_duck \
    RL_DUCK_EVO_RUN=velocity-flat-ppo \
    XBOTICS_FIG_FONT=<TimesSong.ttf> MUJOCO_GL=egl \
      uv run python code/render_evolution.py

可调的都走环境变量：`RL_DUCK_EVO_STAGES`（要几段）、`RL_DUCK_EVO_STEPS`（每段多少步）、
`RL_DUCK_FORCE_CMD`（钉住速度指令，免得某一段恰好采到接近零的指令、看着像没动）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from env import DuckEnv
from PIL import Image, ImageDraw, ImageFont
from render_gallery import _as_uint8, make_policy, result_dir
from rollout import load_policy

# 一段视频里放几档权重，以及每档跑多少步（50 Hz，150 步 = 3 秒）。
STAGES = int(os.environ.get("RL_DUCK_EVO_STAGES") or 6)
STEPS = int(os.environ.get("RL_DUCK_EVO_STEPS") or 150)
# **这段视频必须用固定机位。** 跟拍机位下镜头锁在鸭子身上、地板又是无特征的格子，
# 走得再远也像原地踏步 —— 轮滑那段视频已经栽过一次（实测 5 秒走 1.553 米，
# 录出来看着没动）。而这段片子要给人看的恰恰是"越练走得越远"，
# 位移看不见，这段片子就没有意义。
SIZE = (1280, 960)
DISTANCE = float(os.environ.get("RL_DUCK_EVO_DISTANCE") or 2.6)
ELEVATION = -15.0
LABEL_PX = 44


def checkpoints_of(run_dir: Path) -> list[tuple[int, Path]]:
    """列出一个 run 目录下的全部权重，按迭代数**数值**排序。

    数值排序不是小事：字典序下 `model_400.pt` 排在 `model_2000.pt` 后面，
    接出来的视频就成了乱序的，而画面上每段都有标注、乱序看着像"训练在倒退"。

    Args:
        run_dir: 某次训练的 checkpoint 目录。

    Returns:
        [(迭代数, 路径)]，按迭代数升序。

    Raises:
        SystemExit: 目录里一个 `model_*.pt` 都没有。
    """
    got = []
    for p in run_dir.glob("model_*.pt"):
        m = re.fullmatch(r"model_(\d+)", p.stem)
        if m:
            got.append((int(m.group(1)), p))
    if not got:
        raise SystemExit(f"★ {run_dir} 下没有 model_*.pt")
    return sorted(got)


def pick_stages(all_ckpts: list[tuple[int, Path]], count: int) -> list[tuple[int, Path]]:
    """从全部权重里挑 `count` 档，首尾必取、中间等距。

    首尾必取的理由：这段视频要给人看的就是"最早"和"最后"的差别，
    等距取样如果把最后一档漏掉，最有说服力的那一段就没了。

    Args:
        all_ckpts: `checkpoints_of` 的返回值。
        count: 要挑几档。

    Returns:
        挑出来的 [(迭代数, 路径)]。
    """
    if len(all_ckpts) <= count:
        return all_ckpts
    idx = np.linspace(0, len(all_ckpts) - 1, count).round().astype(int)
    return [all_ckpts[i] for i in sorted(set(idx.tolist()))]


def label_frames(frames: list[np.ndarray], text: str, font_path: str) -> list[np.ndarray]:
    """在每一帧左上角写一行字。

    白字带黑描边，不加半透明底板：底板会盖掉画面，而描边在深色地板和浅色鸭子上都读得出。

    Args:
        frames: 待标注的帧。
        text: 要写的那行字。
        font_path: 字体文件路径（西文 Times New Roman ＋ 中文宋体的合并体）。

    Returns:
        标注后的帧。
    """
    font = ImageFont.truetype(font_path, LABEL_PX)
    out = []
    for f in frames:
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        draw.text((36, 28), text, font=font, fill="white",
                  stroke_width=3, stroke_fill="black")
        out.append(np.asarray(img))
    return out


def main() -> None:
    """渲一段进化视频，落在该任务的结果目录里。

    Raises:
        SystemExit: 权重目录或字体取不到。
    """
    font_path = os.environ.get("XBOTICS_FIG_FONT", "")
    if not font_path or not Path(font_path).is_file():
        raise SystemExit(f"★ 出图字体取不到：{font_path!r}")
    root = Path(os.environ.get("RL_DUCK_CKPT_ROOT") or "")
    run = os.environ.get("RL_DUCK_EVO_RUN") or ""
    if not root.is_dir() or not run:
        raise SystemExit("★ 要给 RL_DUCK_CKPT_ROOT 与 RL_DUCK_EVO_RUN")
    stages: list[tuple[int, Path | None]] = list(pick_stages(checkpoints_of(root / run), STAGES))
    # 开头补一段**没训练过**的随机策略。最早那个 checkpoint 已经是第 200 迭代，
    # 那时它已经会站了 —— 少了这一段，整条视频看下来"进化"并不明显，
    # 而随机策略那三秒是全片对比最强的一段。
    stages.insert(0, (0, None))
    # 末尾再接一档**别的 run** 的权重。为什么需要跨 run：进步全发生在前几百次迭代里，
    # 而正式训练每 200 次才存一档 —— 那一段一张都没留下。于是单独跑了一次只训 400 次、
    # 每 10 次存一档的短训练（`RL_DUCK_SAVE_INTERVAL`）当前半段，
    # 收尾那一档仍取正式训练的最终权重，让"练到头是什么样"也在片子里。
    final = os.environ.get("RL_DUCK_EVO_FINAL")
    if final:
        m = re.fullmatch(r"model_(\d+)", Path(final).stem)
        if not m:
            raise SystemExit(f"★ RL_DUCK_EVO_FINAL 的文件名里读不出迭代数：{final}")
        stages.append((int(m.group(1)), Path(final)))
    print("这段视频要接的档：", ", ".join(str(i) for i, _ in stages))

    frames: list[np.ndarray] = []
    fps = 50.0
    for iteration, ckpt in stages:
        env = DuckEnv(num_envs=1, device="cuda:0", seed=1, render_mode="rgb_array",
                      shadows=False, render_size=SIZE, camera_distance=DISTANCE,
                      camera_elevation=ELEVATION, camera_origin="world",
                      debug_vis=False)
        try:
            model = (make_policy(env, None)[0] if ckpt is None
                     else load_policy(str(ckpt), "cuda:0")[0])
            obs, _critic = env.reset()
            clip: list[np.ndarray] = []
            for _ in range(STEPS):
                with torch.no_grad():
                    actions = model.act_inference(obs)
                obs, _critic, _r, _d, _i = env.step(actions)
                rendered = env.render()
                if rendered is not None:
                    clip.append(_as_uint8(rendered))
            fps = float(env.metadata.get("render_fps", 50.0))
        finally:
            env.close()
        if not clip:
            raise SystemExit(f"★ 第 {iteration} 档一帧都没渲出来")
        label = "训练 0 次（随机策略）" if ckpt is None else f"训练 {iteration} 次迭代"
        frames.extend(label_frames(clip, label, font_path))
        print(f"  {label}：{len(clip)} 帧")

    out = result_dir() / "evolution.mp4"
    media.write_video(str(out), frames, fps=fps)
    print(f"写出 {out}  （{len(stages)} 档、共 {len(frames)} 帧、{len(frames) / fps:.1f} 秒）")


if __name__ == "__main__":
    main()

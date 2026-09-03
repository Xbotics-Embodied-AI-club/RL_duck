"""把交付要的图与视频一次性产出来 —— 交付件的**重出入口**。

为什么要有这么一个入口：`collect_delivery.py` 声明了交付件是什么，
但"怎么产出它"先前散在几份一次性的 shell 脚本与临时命令里。结果是
清单上声明的名字，仓内没有任何入口能产出 —— 照 README 跑一遍必然缺件。

下面 `JOBS` 表是那个"怎么产出"的唯一声明：一个动作一行，写清任务 id、
用哪份权重、单只视频用什么机位、要不要钉住速度指令。改动作就改这张表。

**机位为什么要按动作分**：跟拍机位把位移藏起来 —— 镜头锁在鸭子身上、
地板又是无特征的格子，走得再远也像原地踏步（轮滑那档实测 5 秒走 1.553 米，
录出来看着没动）。所以"要看位移"的动作用世界固定机位、短镜头；
"要看姿态"的动作用跟拍。

跑法（在仓根，需要 GPU；权重取交付目录里那七份）：
    RL_DUCK_CKPT_ROOT=交付/权重 RL_DUCK_RESULT_ROOT=<过程产物根> \
    XBOTICS_FIG_FONT=<TimesSong.ttf> MUJOCO_GL=egl \
      uv run python code/render_delivery.py [动作名 ...]

不给动作名就全做。做完跑 `code/collect_delivery.py` 收件。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 一个动作一行：
#   名字 · 任务 id · 权重相对路径 · 关键帧数 · 关键帧列数 ·
#   单只视频机位（"track" 跟拍 / "fixed" 世界固定）· 单只视频步数 · 钉住的速度指令
JOBS: tuple[dict, ...] = (
    {"name": "走路-v3PPO", "task": "Mjlab-Velocity-Flat-MicroDuck",
     "ckpt": "velocity-flat-ppo/model_6000.pt", "frames": 6, "cols": 3,
     "camera": "track", "steps": 400, "cmd": "", "parallel": True,
     "tile": {"RL_DUCK_COVER_FRACS": "0.25,0.35,0.45"}},
    {"name": "走路-v1REINFORCE", "task": "Mjlab-Velocity-Flat-MicroDuck",
     "ckpt": "velocity-flat-reinforce/model_6000.pt", "frames": 12, "cols": 3,
     "camera": "track", "steps": 400, "cmd": "", "parallel": False,
     "keyframe_steps": 1000},
    {"name": "走路-v2A2C", "task": "Mjlab-Velocity-Flat-MicroDuck",
     "ckpt": "velocity-flat-a2c/model_6000.pt", "frames": 12, "cols": 3,
     "camera": "track", "steps": 400, "cmd": "", "parallel": False,
     "keyframe_steps": 1000},
    {"name": "轮滑", "task": "Mjlab-Velocity-Flat-MicroDuck-Rollers",
     "ckpt": "velocity-flat-rollers-ppo/model_2000.pt", "frames": 6, "cols": 3,
     "camera": "fixed", "steps": 180, "cmd": "lin_vel_x=0.5,lin_vel_y=0,ang_vel_z=0",
     "parallel": True,
     # 轮滑从正面看与站着几乎一样，差别全在腿的弯曲程度上 —— 要侧面机位
     "tile": {"RL_DUCK_COVER_AZIMUTH": "90", "RL_DUCK_COVER_FRACS": "0.45,0.55,0.65"}},
    {"name": "旋转", "task": "Mjlab-Spin-Flat-MicroDuck",
     "ckpt": "spin-flat-ppo/model_2000.pt", "frames": 6, "cols": 3,
     "camera": "track", "steps": 400, "cmd": "", "parallel": True,
     "tile": {"RL_DUCK_COVER_FRACS": "0.45,0.65,0.75"}},
    {"name": "取物", "task": "Mjlab-GroundPick-Flat-MicroDuck",
     "ckpt": "groundpick-flat-ppo/model_2000.pt", "frames": 6, "cols": 3,
     "camera": "track", "steps": 400, "cmd": "", "parallel": True,
     "tile": {"RL_DUCK_COVER_FRACS": "0.55,0.65,0.75"}},
    {"name": "前滚翻", "task": "Mjlab-Roulade-Flat-MicroDuck",
     "ckpt": "roulade-flat-ppo/model_6000.pt", "frames": 12, "cols": 3,
     "camera": "track", "steps": 250, "cmd": "", "parallel": True,
     "keyframe_steps": 230,
     # 翻滚在第 68 步就完成，默认那档相位（0.25–0.75）整段都在它之后
     "tile": {"RL_DUCK_COVER_FRACS": "0.08,0.12,0.16"}},
)

CODE = Path(__file__).resolve().parent
# 六只版：跟拍第一只 + 统一指令（整队才不散）+ 机位钉在 1.8 米。
PARALLEL_ENV = {"RL_DUCK_PARALLEL_TRACK": "1", "RL_DUCK_PARALLEL_RECORD": "400",
                "RL_DUCK_PARALLEL_DISTANCE": "1.8"}


def ckpt_root() -> Path:
    """交付权重的根目录。

    Returns:
        `RL_DUCK_CKPT_ROOT` 指向的目录。

    Raises:
        SystemExit: 没给或者不存在 —— 不去猜一个默认位置，猜错会渲出别档的图。
    """
    root = os.environ.get("RL_DUCK_CKPT_ROOT", "")
    if not root or not Path(root).is_dir():
        raise SystemExit(f"★ RL_DUCK_CKPT_ROOT 要指向交付权重目录，当前 = {root!r}")
    return Path(root)


def run_step(script: str, env: dict[str, str], label: str) -> None:
    """跑一步出图/录像，失败就停下。

    Args:
        script: `code/` 下的脚本名。
        env: 这一步要额外给的环境变量。
        label: 打印用的步骤名。

    Raises:
        SystemExit: 那一步退非零 —— 交付件缺一件就不该继续往下走。
    """
    print(f"── {label}", flush=True)
    merged = {**os.environ, **env}
    rc = subprocess.run([sys.executable, str(CODE / script)], env=merged, check=False).returncode
    if rc:
        raise SystemExit(f"★ {label} 退 {rc}")


def do_job(job: dict) -> None:
    """产出一个动作要的全部交付件：关键帧 / 并行图与六只视频 / 单只视频。

    Args:
        job: `JOBS` 里的一行。
    """
    ckpt = ckpt_root() / job["ckpt"]
    if not ckpt.is_file():
        raise SystemExit(f"★ {job['name']}：权重不在 {ckpt}")
    base = {"RL_DUCK_TASK": job["task"], "RL_DUCK_CHECKPOINT": str(ckpt),
            "RL_DUCK_CKPT_ROOT": str(ckpt_root()),
            "RL_DUCK_KEYFRAME_COUNT": str(job["frames"]),
            "RL_DUCK_KEYFRAME_COLS": str(job["cols"])}
    if job.get("keyframe_steps"):
        base["RL_DUCK_KEYFRAME_STEPS"] = str(job["keyframe_steps"])
    if job["cmd"]:
        base["RL_DUCK_FORCE_CMD"] = job["cmd"]

    run_step("render_gallery.py", {**base, "RL_DUCK_ONLY": "keyframes"},
             f"{job['name']}：关键帧 {job['frames']} 帧 {job['cols']} 列")
    if job["parallel"]:
        run_step("render_gallery.py", {**base, **PARALLEL_ENV, "RL_DUCK_ONLY": "parallel"},
                 f"{job['name']}：六只版并行图与视频")
    video_env = {**base, "RL_DUCK_RUN_NAME": f"交付-{job['name']}",
                 "RL_DUCK_VIDEO_STEPS": str(job["steps"])}
    if job["camera"] == "fixed":
        video_env["RL_DUCK_VIDEO_FIXED"] = "1"
        video_env["RL_DUCK_VIDEO_DISTANCE"] = "2.6"
    run_step("rollout_one.py", video_env, f"{job['name']}：单只视频（{job['camera']} 机位）")
    if job.get("tile"):
        run_step("render_gallery.py", {**base, **job["tile"], "RL_DUCK_ONLY": "tile"},
                 f"{job['name']}：开篇动作带的候选定格")


def main() -> None:
    """按 `JOBS` 产出交付件；命令行给了动作名就只做那几个。"""
    want = set(sys.argv[1:])
    jobs = [j for j in JOBS if not want or j["name"] in want]
    if not jobs:
        raise SystemExit(f"★ 没有匹配的动作。可选：{[j['name'] for j in JOBS]}")
    for job in jobs:
        do_job(job)
    print(f"完成 {len(jobs)} 个动作。接着跑 code/collect_delivery.py 收件；"
          "开篇动作带与三条曲线另有两步，见 交付/README.md。")


if __name__ == "__main__":
    main()

"""出讲义要的三类图：并行仿真图、单动作关键帧序列、开篇那张成品拼图。

三类图共用一条渲染路径（`DuckEnv(render_mode="rgb_array")` 逐帧取画面），区别只在
开几个环境、用哪份权重、怎么拼版：

- **并行仿真图 / 视频**：一屏一群小鸭子各自训练那种，撑视觉冲击。出一张 PNG 加一段 MP4。
  ⚠️ **这张图是叠画出来的，不是一个场地里的实况。** 训练的并行是「多个环境、每只鸭子
  一个环境」—— 每个环境是一份独立的物理世界，鸭子之间既看不见也碰不到。渲染器先画
  第 0 个环境，再把邻近几个环境的鸭子的几何体**追加**进同一张画面（`max_extra_envs`），
  所以图注必须说清这一点，否则读者会以为它们在互相避让。
- **关键帧序列**：一个动作横排 4–6 帧，读者一眼看出它做了什么。
- **开篇拼图**：把已经渲好的各动作关键帧摆成一个网格，先给结果再讲道理。

没有权重也能跑：`CHECKPOINT` 留空就用随机初始化的策略渲，用来先验证版面对不对 ——
这一步值得在开训之前做一次，框歪了当场就能改，不必等训完再重渲一轮。

讲义对应：导言（图 14-1 / 14-2）、4.8 节（并行仿真）、8.2–8.6 节（各动作关键帧）。

跑法（在仓根）：
    uv run python code/render_gallery.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import mediapy as media
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TASK, DuckEnv, task_slug  # noqa: E402
from model import ActorCritic  # noqa: E402
from plot_reward_curves import use_project_font  # noqa: E402
from rollout import load_policy  # noqa: E402

# ======================== 这一轮渲什么 ========================
# 权重文件。留 None 就自动取当前 TASK 的 PPO 那一档**最新**的 checkpoint；
# 一个都没有（还没开训）就退回随机初始化的策略，只为验证版面 ——
# 那种图文件名带 untrained 后缀，别拿它当结果图。
# 想钉某一个具体的 checkpoint，把路径写在这里。
CHECKPOINT: str | None = None

# 渲染分辨率。mjlab 的默认是 320×240，够在窗口里看、不够印进讲义 ——
# 必须显式放大，否则图照样存得下来、脚本照样退 0，只是印在纸上是一团马赛克。
PARALLEL_SIZE = (1600, 900)
KEYFRAME_SIZE = (960, 720)

# 相机的量，实测比出来的。三个量一起调才有用：默认视线几乎水平，
# 地平线落在画面中间、上面三四成是纯黑的天；而 lookat 钉在地面原点，
# 鸭子沉在画面下缘。
#   单只：0.8 米 / 俯 15 度，视线抬到躯干高度，关节和脚都看得清。
#   一群：**只渲 6 只，而且把它们摆得挨近**。
#        踩过两次坑才对：32 只那版每只都是小点；改成 6 只之后仍然小 ——
#        因为我退相机去"装下"它们，而任务默认的环境间距本来就有十几米宽。
#        正确的杠杆是**改间距**，不是退相机：每个环境是独立的物理世界，
#        间距只影响画面偏移与地形分片查询，平地上纯属视觉安排。
PARALLEL_SPACING = 0.8
PARALLEL_ELEVATION = -15.0
KEYFRAME_DISTANCE = 0.8
KEYFRAME_ELEVATION = -15.0
KEYFRAME_LOOKAT = (0.0, 0.0, 0.18)

# 并行仿真图开多少个环境（全都画进画面）。
# 六只：既表达了"不止一只、各自在学"，每只又还看得清在做什么。
PARALLEL_ENVS = 6
# 先跑一会儿让姿态散开，别都停在初始站姿；但也别跑太久 ——
# 它们会各自走开，队形散了就框不住了。
PARALLEL_WARMUP_STEPS = 90

# 关键帧序列：横排几帧、总共跑多少步、从第几步开始取。
KEYFRAME_COUNT = 5
KEYFRAME_STEPS = 260
KEYFRAME_SKIP = 60  # 前若干步是 reset 后的过渡，动作还没起来

# 开篇拼图：按这个顺序摆格子。每个元素是一个 task slug，
# 对应 `result/<slug>/keyframes.png` 必须已经渲好。缺哪个就跳过哪个。
COVER_LAYOUT = (
    "velocity-flat",
    "velocity-flat-rollers",
    "standup-flat",
    "spin-flat",
    "ballkick-flat",
    "sitstand-flat",
)
COVER_COLUMNS = 2
# ==============================================================


def latest_checkpoint() -> str | None:
    """给出当前 TASK 的 PPO 那一档最新的 checkpoint 路径。

    自动取而不是写死一个迭代号：训练还在跑的时候，"最新" 每二十分钟就换一个，
    写死的号码只会让人每次手改一遍、并且早晚忘记改。

    Returns:
        最新 checkpoint 的路径；那一档还没有任何 checkpoint 时返回 None。
    """
    if CHECKPOINT:
        return CHECKPOINT
    root = os.environ.get("RL_DUCK_CKPT_ROOT")
    base = Path(root) if root else Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "rl_duck"
    run_dir = base / f"{task_slug()}-ppo"
    found = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return str(found[-1]) if found else None


def parallel_camera(num_envs: int) -> tuple[float, tuple[float, float, float]]:
    """按这批环境实际铺开的范围，算出相机该放多远、看向哪。

    先空建一次环境（不带渲染器，便宜）把真实的环境原点读出来，再取中心与跨度 ——
    而不是按环境间距猜数。猜的数在 `PARALLEL_ENVS` 一改就悄悄偏掉：镜头照样出图、
    脚本照样退 0，只是鸭子跑到画面角上或者小成一排点。

    Args:
        num_envs: 这张图要开多少个环境。

    Returns:
        (相机距离, 视线目标)。视线的 z 抬到鸭子躯干高度，不钉在地面。
    """
    probe = DuckEnv(num_envs=num_envs, device="cuda:0", seed=0, render_mode=None,
                    env_spacing=PARALLEL_SPACING)
    try:
        origins = probe.unwrapped.scene.env_origins[:num_envs].float()
        center = origins.mean(dim=0).tolist()
        span = float((origins.max(dim=0).values - origins.min(dim=0).values)[:2].max())
    finally:
        probe.close()
    # 间距已经压小了，这里只需给一点余量：跨度 + 1.3 米。
    # 竖直视角约 45 度，2.5 米处画面高约 2 米、宽约 3.6 米，这个跨度装得下还有边。
    distance = span * 0.8 + 1.0
    lookat = (float(center[0]), float(center[1]), 0.22)
    print(f"一群那张图：{num_envs} 只，间距 {PARALLEL_SPACING} 米、实际铺开 {span:.2f} 米"
          f" => 相机 {distance:.2f} 米，看向 {lookat}")
    return distance, lookat


def result_dir(slug: str | None = None) -> Path:
    """某个任务的结果目录，图和 json 都落在这里。

    Args:
        slug: 任务短名，默认取当前 `TASK` 的。

    Returns:
        `result/<slug>/` 的绝对路径，已建好。
    """
    d = Path(__file__).resolve().parents[1] / "result" / (slug or task_slug())
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_policy(env: DuckEnv, checkpoint: str | None):
    """给出一个能出动作的策略：有权重就加载，没有就随机初始化。

    随机策略只用于验证版面 —— 它渲出来的画面不是结果，图注里不能当成结果写。

    Args:
        env: 已经构好的环境，用来取三个维度。
        checkpoint: 权重路径，None 表示随机初始化。

    Returns:
        (策略, 说明字符串)，说明字符串写进文件名后缀，免得把验证图当结果图。
    """
    if checkpoint:
        model, iteration = load_policy(checkpoint, env.device)
        return model, f"iter{iteration}"
    model = ActorCritic(
        obs_dim=env.obs_dim, critic_obs_dim=env.critic_obs_dim, action_dim=env.action_dim
    )
    model.eval()
    model.to(env.device)
    return model, "untrained"


def _as_uint8(frame) -> np.ndarray:
    """把 render 回来的东西统一成一张 uint8 的 RGB 图。

    mjlab 在不同配置下可能回一批画面（4 维）或一张（3 维）、浮点或整型，
    这里一次性抹平，后面的拼版代码就只需要处理一种形状。
    """
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame


def render_parallel(checkpoint: str | None = None) -> Path:
    """渲一张「一屏一群小鸭子」的并行仿真图。

    Args:
        checkpoint: 权重路径，None 用随机策略（只验证版面）。

    Returns:
        写出的 PNG 路径。
    """
    distance, lookat = parallel_camera(PARALLEL_ENVS)
    env = DuckEnv(
        num_envs=PARALLEL_ENVS,
        device="cuda:0",
        seed=0,
        render_mode="rgb_array",
        max_extra_envs=PARALLEL_ENVS - 1,
        env_spacing=PARALLEL_SPACING,
        render_size=PARALLEL_SIZE,
        camera_distance=distance,
        camera_elevation=PARALLEL_ELEVATION,
        camera_lookat=lookat,
    )
    frames: list[np.ndarray] = []
    try:
        model, tag = make_policy(env, checkpoint)
        obs, _critic = env.reset()
        for _ in range(PARALLEL_WARMUP_STEPS):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic, _r, _d, _i = env.step(actions)
            rendered = env.render()
            if rendered is not None:
                frames.append(_as_uint8(rendered))
        fps = float(env.metadata.get("render_fps", 50.0))
    finally:
        env.close()

    if not frames:
        raise RuntimeError("render() 一帧都没回 —— 检查 render_mode 与 viewer 配置")
    out = result_dir() / f"parallel-{tag}.png"
    plt.imsave(out, frames[-1])
    video = result_dir() / f"parallel-{tag}.mp4"
    media.write_video(str(video), frames, fps=fps)
    h, w = frames[-1].shape[:2]
    print(f"写出 {out} 与 {video}  ({w}×{h}，{PARALLEL_ENVS} 个环境，{len(frames)} 帧)")
    return out


def render_keyframes(checkpoint: str | None = None) -> Path:
    """渲一条横排的关键帧序列，读者一眼看出这个动作做了什么。

    帧是从一段确定性 rollout 里**等间隔**取的，不是挑好看的那几帧 ——
    挑帧会让图说的话比策略实际做到的多。

    Args:
        checkpoint: 权重路径，None 用随机策略（只验证版面）。

    Returns:
        写出的 PNG 路径。
    """
    env = DuckEnv(
        num_envs=1,
        device="cuda:0",
        seed=1,
        render_mode="rgb_array",
        render_size=KEYFRAME_SIZE,
        camera_distance=KEYFRAME_DISTANCE,
        camera_elevation=KEYFRAME_ELEVATION,
        camera_lookat=KEYFRAME_LOOKAT,
    )
    frames: list[np.ndarray] = []
    try:
        model, tag = make_policy(env, checkpoint)
        obs, _critic = env.reset()
        for _ in range(KEYFRAME_STEPS):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic, _r, _d, _i = env.step(actions)
            rendered = env.render()
            if rendered is not None:
                frames.append(_as_uint8(rendered))
    finally:
        env.close()

    usable = frames[KEYFRAME_SKIP:] or frames
    if not usable:
        raise RuntimeError("render() 一帧都没回 —— 检查 render_mode 与 viewer 配置")
    picks = np.linspace(0, len(usable) - 1, KEYFRAME_COUNT).round().astype(int)
    strip = [usable[i] for i in picks]

    use_project_font()
    fig, axes = plt.subplots(1, KEYFRAME_COUNT, figsize=(2.3 * KEYFRAME_COUNT, 2.6), dpi=200)
    for ax, img, idx in zip(np.atleast_1d(axes), strip, picks, strict=True):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"第 {int(idx) + KEYFRAME_SKIP} 步", fontsize=8)
    fig.suptitle(TASK.removeprefix("Mjlab-").replace("-MicroDuck", ""), fontsize=10)
    fig.tight_layout()
    out = result_dir() / f"keyframes-{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"写出 {out}  (取了 {KEYFRAME_COUNT} 帧，共 {len(usable)} 帧可选)")
    return out


def build_cover() -> Path | None:
    """把各动作已渲好的关键帧摆成开篇那张拼图。

    只读已有的 PNG，不重新渲 —— 所以它可以在每训完一个动作之后重跑一次，
    格子会一个个填上。

    Returns:
        写出的 PNG 路径；一张都凑不出来时返回 None。
    """
    root = Path(__file__).resolve().parents[1] / "result"
    tiles: list[tuple[str, np.ndarray]] = []
    for slug in COVER_LAYOUT:
        found = sorted((root / slug).glob("keyframes-*.png")) if (root / slug).is_dir() else []
        if not found:
            print(f"拼图缺一格：{slug}（还没渲关键帧）")
            continue
        tiles.append((slug, plt.imread(found[-1])))
    if not tiles:
        print("一格都凑不出来，先渲各动作的关键帧。")
        return None

    use_project_font()
    rows = -(-len(tiles) // COVER_COLUMNS)
    fig, axes = plt.subplots(rows, COVER_COLUMNS, figsize=(6.6 * COVER_COLUMNS, 2.2 * rows), dpi=200)
    flat = np.atleast_1d(axes).ravel()
    for ax, (slug, img) in zip(flat, tiles, strict=False):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(slug, fontsize=9)
    for ax in flat[len(tiles):]:
        ax.axis("off")
    fig.suptitle("这里没有一个动作是人教的", fontsize=13)
    fig.tight_layout()
    out = root / "cover.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"写出 {out}  ({len(tiles)} 格)")
    return out


def main():
    """当前 `TASK` 的并行仿真图 + 关键帧序列，然后重拼一次开篇图。"""
    ckpt = latest_checkpoint()
    print(f"TASK = {TASK}，权重 = {ckpt or '（一个 checkpoint 都没有，用随机初始化，仅验证版面）'}")
    render_parallel(ckpt)
    render_keyframes(ckpt)
    build_cover()


if __name__ == "__main__":
    main()

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
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TASK, DuckEnv, task_slug
from model import ActorCritic
from plot_reward_curves import use_project_font
from rollout import load_policy

# ======================== 这一轮渲什么 ========================
# 权重文件。留空就自动取当前 TASK 的 PPO 那一档**最新**的 checkpoint；
# 一个都没有（还没开训）就退回随机初始化的策略，只为验证版面 ——
# 那种图文件名带 untrained 后缀，别拿它当结果图。
#
# 批量出图时用 RL_DUCK_CHECKPOINT 按进程指定：**最终 checkpoint 不等于最好的**
# （实测摔倒起身那档末段比峰值段低 0.045），出视频要挑峰值段那个权重。
CHECKPOINT: str | None = os.environ.get("RL_DUCK_CHECKPOINT") or None

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
PARALLEL_ELEVATION = -22.0
KEYFRAME_DISTANCE = 0.8
KEYFRAME_ELEVATION = -15.0
KEYFRAME_LOOKAT = (0.0, 0.0, 0.18)

# 并行仿真图开多少个环境（全都画进画面）。
# 六只：既表达了"不止一只、各自在学"，每只又还看得清在做什么。
PARALLEL_ENVS = 6
# 预热与录制**必须分开两个数**。之前它们是同一个：为了让队形别散，我把这个数
# 从 150 砍到 45，顺手把视频从 3 秒砍成了 0.9 秒 —— 一个常量管着两件事。
# 预热只是等步态起来（一个步态周期约 20 步），录制才决定视频多长。
# ⚠️ 上限来自任务本身，实测过：六只各拿一个**不同的**随机速度指令，所以它们必然走散。
# 200 帧（4 秒）跑完只剩 3 只还在画面里 —— 我先前以为 4 秒还框得住，量了才知道不行。
# 预热 12 步（约半个步态周期）让它们不是僵在初始站姿；录 1000 帧（20 秒）。
# 长镜头是安全的，两条实测支撑：160 步里最远两只从 2.50 米收到 2.03 米（往中间聚、
# 不是散开），群体中心只漂 13 厘米；而且相机是跟拍的（track_robot），不是固定机位。
PARALLEL_PREROLL = 12
PARALLEL_RECORD = 1000

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


def parallel_camera(num_envs: int, checkpoint: str | None) -> tuple[float, float, tuple[float, float, float]]:
    """先跑一遍量出这几只鸭子的**实际**位置，再据此算相机。

    为什么不用出生点（`env_origins`）算：实测过，两者差 45%。出生点是 3×2 的格子、
    x 方向跨 1.6 米，而机器人加上各自的初始姿态偏移之后，实际 x 跨度是 2.50 米。
    按出生点算余量必然框不住。

    长镜头是安全的：实测 160 步里最远两只的距离从 2.50 米收到 2.03 米（六只各追各的
    速度指令，净效果是往中间聚而不是散开），群体中心只漂 13 厘米。

    Args:
        num_envs: 这张图要开多少个环境。
        checkpoint: 权重路径；None 就用随机策略量（位置分布差不多）。

    Returns:
        (相机距离, 方位角, 视线目标)。
    """
    probe = DuckEnv(num_envs=num_envs, device="cuda:0", seed=0, render_mode=None,
                    env_spacing=PARALLEL_SPACING)
    try:
        model, _tag = make_policy(probe, checkpoint)
        obs, _critic = probe.reset()
        for _ in range(PARALLEL_PREROLL):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic, _r, _d, _i = probe.step(actions)
        pos = _root_xy(probe, num_envs)
    finally:
        probe.close()

    center = pos.mean(axis=0)
    centered = pos - center
    if num_envs > 1:
        # 主轴 = 这几只实际排开的方向（协方差第一主成分）。
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        along = centered @ axis
        across = centered @ np.array([-axis[1], axis[0]])
        long_extent = float(along.max() - along.min())
        short_extent = float(across.max() - across.min())
    else:
        axis = np.array([1.0, 0.0])
        long_extent = short_extent = 0.0

    # MuJoCo 的 azimuth 是绕竖直轴的角度（度）；取主轴方向再转 90°，
    # 相机就从垂直于主轴的方向看过去，那条线在画面里横着铺满。
    azimuth = float(np.degrees(np.arctan2(axis[1], axis[0])) + 90.0)
    # 竖直视角约 45 度、16:9，所以水平视野 ≈ 2·d·tan(38°) ≈ 1.56·d。
    # 要让主轴跨度只占画面宽的 70%（两边留边），d ≈ 跨度 / (1.56 × 0.7)。
    # 再加上纵深那一档的一半，免得靠后的那几只被上边缘切掉。
    distance = long_extent / 1.09 + short_extent * 0.5 + 0.6
    lookat = (float(center[0]), float(center[1]), 0.22)
    print(f"一群那张图：{num_envs} 只，实测主轴跨 {long_extent:.2f} 米、纵深 {short_extent:.2f} 米"
          f" => 相机 {distance:.2f} 米 / 方位 {azimuth:.0f}° / 俯 {PARALLEL_ELEVATION}° / 看向 {lookat}")
    return distance, azimuth, lookat


def _root_xy(env: DuckEnv, num: int) -> np.ndarray:
    """取每个环境里机器人根节点的世界坐标 xy。

    属性名在 mjlab 各版本里叫法不同，按可用的取；一个都没有就把候选列出来报错 ——
    不猜、不回落，猜错会静默给出一组错坐标，比报错难查得多。
    """
    data = env.unwrapped.scene["robot"].data
    for name in ("root_link_pos_w", "root_com_pos_w"):
        v = getattr(data, name, None)
        if v is not None:
            return v.float().cpu().numpy()[:num, :2]
    raise AttributeError(
        f"找不到根节点世界坐标；可用属性：{[a for a in dir(data) if not a.startswith('_')]}"
    )


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
    distance, azimuth, lookat = parallel_camera(PARALLEL_ENVS, checkpoint)
    env = DuckEnv(
        num_envs=PARALLEL_ENVS,
        device="cuda:0",
        seed=0,
        render_mode="rgb_array",
        max_extra_envs=PARALLEL_ENVS - 1,
        env_spacing=PARALLEL_SPACING,
        shadows=False,
        # 用世界固定相机对准群体中心，而不是跟拍某一只：跟拍会把被跟的那只放在
        # 画面正中，而它在群体的一端，于是其余五只全挤到一侧。群体中心 160 步只漂
        # 13 厘米，固定机位足够稳。
        camera_origin="world",
        render_size=PARALLEL_SIZE,
        camera_distance=distance,
        camera_elevation=PARALLEL_ELEVATION,
        camera_azimuth=azimuth,
        camera_lookat=lookat,
    )
    frames: list[np.ndarray] = []
    try:
        model, tag = make_policy(env, checkpoint)
        obs, _critic = env.reset()
        # 预热：只推进仿真、不取画面，等步态起来。
        for _ in range(PARALLEL_PREROLL):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic, _r, _d, _i = env.step(actions)
        # 录制：这一段才是视频。
        for _ in range(PARALLEL_RECORD):
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
    # 静帧取**第一帧**而不是最后一帧：预热之后步态已经起来、六只还都在画面里；
    # 到最后一帧它们已经各自走开，那张图只剩一半。
    plt.imsave(out, frames[0])
    video = result_dir() / f"parallel-{tag}.mp4"
    media.write_video(str(video), frames, fps=fps)
    h, w = frames[0].shape[:2]
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
        shadows=False,
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

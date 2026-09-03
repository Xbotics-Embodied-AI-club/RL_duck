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
import re
import sys
from pathlib import Path

import matplotlib
import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TASK, DuckEnv, result_base, task_slug
from model import ActorCritic
from plot_reward_curves import use_project_font
from rollout import align_curriculum, check_task_match, load_policy

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

# 相机的量，实测比出来的。默认视线几乎水平，地平线落在画面中间、上面三四成是纯黑的天，
# 所以距离与俯角必须显式给。
#   单只：0.8 米 / 俯 15 度，跟拍躯干（`camera_origin="asset_root"`）。
#        **这一档没有 lookat**：跟拍模式下 MuJoCo 每帧用被跟刚体的位置覆盖注视点，
#        配置里的 lookat 是死值（`env.py` 里那条实测：平移 lookat 1.12/−0.56 米重渲，
#        两张图逐像素相同）。跟拍本身就把鸭子放在画面中央，抬视线这件事不需要 lookat。
#   一群：**只渲 6 只，而且把它们摆得挨近**。环境默认间距十几米，退相机去"装下"它们
#        只会让每只都成小点（32 只那版、6 只那版都实测过）。正确的杠杆是改间距：
#        每个环境是独立的物理世界，间距只影响画面偏移与地形分片查询，平地上纯属视觉安排。
#        这一档用世界固定相机 + lookat 对准群体中心，lookat 在那里才真的起作用。
PARALLEL_SPACING = 0.8
PARALLEL_ELEVATION = -22.0
KEYFRAME_DISTANCE = 0.8
KEYFRAME_ELEVATION = -15.0

# 并行仿真图开多少个环境（全都画进画面）。
# 六只：既表达了"不止一只、各自在学"，每只又还看得清在做什么。
PARALLEL_ENVS = 6
# 预热与录制**必须分开两个数**。之前它们是同一个：为了让队形别散，把这个数
# 从 150 砍到 45，顺手把视频从 3 秒砍成了 0.9 秒 —— 一个常量管着两件事。
# 预热只是等步态起来（一个步态周期约 20 步），录制才决定视频多长。
# ⚠️ 这一档是**世界固定相机**（见 render_parallel 里的 camera_origin="world"），不是跟拍。
# 固定机位下能录多长由"六只走不走散"决定，而它们必然走散 —— 每只拿一个**不同的**随机
# 速度指令。实测的上限是 200 帧（4 秒）：跑完只剩 3 只还在画面里。
# 现在给的 1000 帧（20 秒）**超过那个上限**，所以视频后段基本是空地板。
# 这是有意接受的：静帧取的是 frames[0]（`render_parallel` 里），不受影响；
# 那段 mp4 只当"很多只在各自跑"的动态素材，不要求全程六只都在框内。
# 要一段全程满画面的视频就把它改成 200。
PARALLEL_PREROLL = 12
# 录多少帧。交付用的六只版按需缩短：整队一起走的话二十秒会走出十米。
PARALLEL_RECORD = int(os.environ.get('RL_DUCK_PARALLEL_RECORD') or 1000)

# 关键帧序列：横排几帧。取帧的**窗口**不是常量，按任务的回合长度算（见 keyframe_window）。
# 取 6 帧、排成两行三列（见 render_keyframes 里为什么不横排一行）。
# 6 而不是 5：5 帧在三列网格里会空出一格，而 6 帧刚好填满两行。
#
# ⚠️ **6 帧是给讲义排版用的，不一定够用来判定动作有没有出现。**
# 短回合任务（前滚翻 250 步、起身 300 步）按 6 帧取样，相邻帧间隔 40–50 步；
# 而一个前滚翻在 50 赫兹下大概只要 30–60 步 —— **整个动作可能落在两帧之间，看不见。**
# 判定用的探针渲图应当加密取样：`RL_DUCK_KEYFRAME_COUNT=12` 按进程覆盖它
# （12 帧配 4 列是三行，版面照样成立）。讲义里用的图仍取默认的 6。
_DEFAULT_KEYFRAME_COUNT = 6
KEYFRAME_COUNT = int(os.environ.get("RL_DUCK_KEYFRAME_COUNT") or _DEFAULT_KEYFRAME_COUNT)
KEYFRAME_COLS = int(os.environ.get("RL_DUCK_KEYFRAME_COLS") or 3)
# 长回合、周期性动作（走路）的取帧窗口：跳掉启动瞬态，再取一段。
# 采样窗口长度。可按进程覆盖：判「摔没摔」要覆盖整个回合（走路那档 1000 步），
# 而讲义里展示动作用的是 260 步这档窗口 —— 两种用途要的窗口长度不一样。
KEYFRAME_STEPS = int(os.environ.get('RL_DUCK_KEYFRAME_STEPS') or 260)
# 有没有显式给过窗口长度 —— 短回合任务只在给过的时候才截短。
_STEPS_GIVEN = bool(os.environ.get('RL_DUCK_KEYFRAME_STEPS'))
KEYFRAME_SKIP = 60
# 判定"这是 episodic 任务还是周期性任务"的阈值，单位是**秒**。
# 它必须与 KEYFRAME_STEPS 分开，而且不能用步数表达 —— 两条都是实测踩出来的：
#   · 一个常量管两件事：先前用 KEYFRAME_STEPS 兼作阈值，改绘图长度就会改掉任务分类。
#   · 用步数当阈值判错了：260 步的阈值把起身的 300 步（6 秒）回合判成"长回合"，
#     于是那个 0.3 秒就完成的起身动作照旧落在窗口外，五帧仍然逐帧相同。
# 本任务族的实际分布：走路 20 秒；起身 6 秒、坐站 6 秒、前滚翻 5 秒、取物与踢球同量级。
# 10 秒落在这两簇中间，两边都留着一倍余量。
EPISODIC_THRESHOLD_SECONDS = 10.0

# 开篇拼图：按这个顺序摆格子。每个元素是一个 task slug，
# 对应 `result/<slug>/keyframes.png` 必须已经渲好。缺哪个就跳过哪个。
#
# **这张表只放看图确认学成了的动作。** 讲义开篇是"先给结果"，摆一个其实没学成的
# 动作进去，等于开篇就在骗人。
# 不在这里的两档，各有各的判定依据（都是看图判的，不是看奖励数字）：
#   · 踢球：课程表末档第 1500 迭代、我们给了 2000（预算够），画面上鸭子侧倒、
#     球没动 ⇒ 真没学成（bd xb-8ffo）。
#   · 坐立：跑到自己课程表 88%（末档 2500）仍六帧十二帧都一动不动，已停训（bd xb-zucf）。
#   · 起身：按配方跑满 6000（超末档 4000 两千），3200/4000/6000 三个判定点画面完全相同，
#     只到"把躯干撑起约 45 度并维持"，从不站起（bd xb-p4c3）。
# 这六格正好是六个看图确认成功的动作。**缺项会被自动跳过**，
# 所以万一某一格的关键帧还没渲，拼图不会失败，只会少一格 —— 建完记得数格子。
# 每格：(task slug, 讲义里那个动作的中文名)。中文名写在这里而不是用 slug ——
# 开篇拼图是给读者看的，`velocity-flat-rollers` 对读者不是信息。
COVER_LAYOUT = (
    ("velocity-flat", "走路"),
    ("velocity-flat-rollers", "轮滑"),
    ("spin-flat", "原地旋转"),
    ("groundpick-flat", "嘴尖取物"),
    ("roulade-flat", "前滚翻"),
)
COVER_COLUMNS = 3
# 动作带的版面：一行一个动作，行内三张是同一个动作的三个瞬间。
# 左边留一条写动作名的空白，图上不烧任何字。
_COVER_TILE_PX = (560, 420)
_COVER_LABEL_PX = 280
_COVER_GUTTER_PX = 10
# 开篇拼图每格的候选取样点（占回合的比例）。开头是启动瞬态、末尾常常已经倒了，
# 所以只在中间那段里取；一个动作在回合里反复出现，不同相位看着差别很大，
# 所以多给几张让人挑，而不是让脚本替人决定哪张"典型"。
# crop_to_content 的两个量：池化块边长、以及算作有内容的亮度阈值。
# 块取 12：格线约 1–2 像素宽，在 12×12 块里被摊薄到阈值以下；再大会把鸭子细部也摊掉。
# 开篇定格的机位距离（米）。比 KEYFRAME_DISTANCE 远，为的是构图留白。
_COVER_TILE_DISTANCE = 0.95

_BLOCK = 12
# 帧边缘那条**纯黑**带（渲染缓冲区没画到的部分）：实测是精确的 0.000，
# 而地板最暗处 0.246 —— 0.05 这条线中间空着五倍余量，不是猜的。
_DARK_LEVEL = 0.05
# 最多削掉一条边的这个比例，免得判据出错时把整张图削光。
# **0.15 太紧**：实测帧顶那条黑带占 16–17%，卡在上限上削不完，留下一条更细的黑边 ——
# 比不削更难看，因为看上去像是故意留的。0.30 仍然安全：地板最暗处 0.246，
# 离 0.05 那条线差着五倍，不存在把地板当黑边削掉的可能。
_DARK_MAX_FRAC = 0.30
# 阈值 0.6 是量出来的，不是猜的：在一张定格上按 8/12/16 三种块大小扫过
# 0.45 / 0.55 / 0.6 / 0.7 四档，0.45 时命中块横跨整幅 65 行里的 2–64 行
# （地板格线被算成了内容），0.6 收到 2–32 行 —— 那才是鸭子占的那一块。
_CONTENT_LEVEL = 0.6
# 主体到裁框上下沿至少留主体高度的这个比例 —— 别让头顶到边。
_MIN_HEADROOM = 0.22
# 裁框至少保留原图高度的这个比例 —— 免得裁出一张分辨率不够的小图。
_MIN_KEEP = 0.62

_COVER_TILE_FRACS = tuple(
    float(x) for x in os.environ.get("RL_DUCK_COVER_FRACS", "0.25,0.35,0.45,0.55,0.65,0.75").split(",")
)
# ⚠️ 默认那组只覆盖回合的中后段，**对"动作发生在很早"的任务是盲的**：
# 前滚翻整个翻滚在第 0–68 步就完成了，而回合有 250 步 —— 0.25 已经是第 62 步之后，
# 六张候选没有一张拍到倒立那一刻。这类任务要用 RL_DUCK_COVER_FRACS 显式往前挪。
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
    # 竖直视角 45 度、16:9 ⇒ 水平半角 = atan(tan22.5° × 16/9) = 36.4°，
    # 所以画面在距离 d 处的实际宽度 ≈ 2·d·tan(36.4°) = 1.475·d。
    #
    # 余量**全部按比例给**，不再加常数项。先前那版写的是
    # `long_extent / 1.09 + short_extent * 0.5 + 0.6`，两个加数把距离整体推远了六成：
    # 旋转那张图主轴只跨 1.6 米，按 80% 填充该站 1.36 米，那版算出 2.47 米，
    # 于是六只鸭子只占画面宽的一半、下面一大片空地板。**这就是"渲得太小"的成因** ——
    # 常数余量在跨度小的时候占比最大，而跨度小恰恰是原地动作（旋转、起身）的常态。
    #
    # 纵深项保留但只给 0.35 倍：靠后那几只需要一点后仰余量，但它不该按 1:1 吃掉画面。
    #
    # 下限 0.8 米是必需的，不是保险：`num_envs == 1` 时两个跨度都是 0，
    # 纯比例式会算出 0 —— 相机落在鸭子体内，渲出来是一片糊。0.8 与单只关键帧
    # 那档（KEYFRAME_DISTANCE）取同一个值，两种图的观感才一致。
    distance = max(long_extent / (1.475 * 0.80) + short_extent * 0.35, KEYFRAME_DISTANCE)
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
    d = result_base() / (slug or task_slug())
    d.mkdir(parents=True, exist_ok=True)
    return d


def keyframe_window(env: DuckEnv) -> tuple[int, int]:
    """按任务的回合长度算出关键帧该在哪一段里取。

    **这里曾经是一个真正让人看不见结果的坑。** 原先窗口写死成「跳过前 60 步、
    在 260 步里取 5 帧」—— 那是按**走路**定的：走路的回合 20 秒（1000 步），
    前 60 步是从初始站姿起步的瞬态，跳掉正好。

    但起身、坐站、前滚翻这些是 episodic 任务，回合只有 5–6 秒（250–300 步），
    而那个动作**整个发生在前 60 步里**（实测：起身在第 15 步就已经站直）。
    按走路的窗口渲，5 帧全落在动作结束之后，画面上是一只一动不动趴着的鸭子 ——
    而图照样出、脚本照样退 0。「动作没出现」有一部分是量具看不见它。

    判据只有一条，而且是关于**任务性质**的：回合短（episodic，动作就是这一个回合）
    就从第 0 步把整段看完；回合长（周期性动作，回合只是个时长上限）才跳掉启动瞬态。

    Args:
        env: 已建好的环境。

    Returns:
        (跳过多少步, 总共跑多少步)。
    """
    if env.episode_seconds <= EPISODIC_THRESHOLD_SECONDS:
        # 短回合默认整段看完；给了 `RL_DUCK_KEYFRAME_STEPS` 就当上界用 ——
        # 想只看回合的前一段时（比如动作在第 68 步就完成、后面是另一回事），
        # 得有办法把窗口截短，而不是被迫连回合最后一帧一起摆上去。
        return 0, min(env.episode_steps, KEYFRAME_STEPS) if _STEPS_GIVEN else env.episode_steps
    return KEYFRAME_SKIP, KEYFRAME_STEPS


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
        model, iteration, settings = load_policy(checkpoint, env.device)
        # 与 rollout 同一口径，两件事都要：
        # ① 核对这份权重是不是当前任务训的 —— 十八个任务的观测/动作维度全相同，
        #    错配不会报形状错，只会渲出一段"这个动作没学会"的画面并覆盖真结果。
        #    批量出图时一台机器七个进程各带一份 TASK 与 CHECKPOINT，错配一次就说不清了。
        # ② 把课程表对齐到这份权重训练到的那一档，否则渲的是「第 0 档课程下的初始姿态」，
        #    不是这份权重实际面对的分布。
        check_task_match(env, settings, checkpoint)
        align_curriculum(env, iteration, settings)
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
    # 机位可按进程钉住。自动算出来的距离是按**当前**群体外接尺寸定的，
    # 而跟拍二十秒之后各只已经散开，于是取物那档算出来的距离远到六只只剩几十像素。
    # 交付用的六只版直接给一个固定距离：散出画面的那只就让它出去，
    # 留在画面里的那几只要看得清。
    forced_distance = os.environ.get("RL_DUCK_PARALLEL_DISTANCE")
    if forced_distance:
        distance = float(forced_distance)
    # 交付用的六只版要跟拍。**固定机位配上统一指令会走出画面**：六只拿到同一条
    # 速度指令之后整队一起走，0.5 米/秒 跑二十秒就是十米，画面里只剩空地板。
    # 跟拍第一只即可 —— 整队同速，队形不散。
    track = bool(os.environ.get("RL_DUCK_PARALLEL_TRACK"))
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
        camera_origin="asset_root" if track else "world",
        debug_vis=False,  # 同上
        render_size=PARALLEL_SIZE,
        camera_distance=distance,
        camera_elevation=PARALLEL_ELEVATION,
        camera_azimuth=azimuth,
        # lookat 只在自由相机下生效，跟拍模式传它会被 DuckEnv 拒掉。
        camera_lookat=None if track else lookat,
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
        # 显式跟拍躯干。不给这个参数时 mjlab 走默认的 AUTO（跟拍第一个非固定刚体），
        # 画面恰好也对 —— 于是"取景到底是我们定的还是撞上的"分不出来。
        # 一份教学材料里，参数生效与不生效必须看得出来。
        camera_origin="asset_root",
        debug_vis=False,  # 关掉指令箭头，见 env.py 那段注释
    )
    frames: list[np.ndarray] = []
    try:
        model, tag = make_policy(env, checkpoint)
        skip, steps = keyframe_window(env)
        print(
            f"关键帧窗口：跳过 {skip} 步、跑 {steps} 步"
            f"（回合 {env.episode_seconds:.1f} 秒 = {env.episode_steps} 步，"
            f"episodic 阈值 {EPISODIC_THRESHOLD_SECONDS:.0f} 秒）"
        )
        obs, _critic = env.reset()
        for _ in range(steps):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic, _r, _d, _i = env.step(actions)
            rendered = env.render()
            if rendered is not None:
                frames.append(_as_uint8(rendered))
    finally:
        env.close()

    if not frames:
        raise RuntimeError("render() 一帧都没回 —— 检查 render_mode 与 viewer 配置")
    # 跳帧与标签必须用**同一个** offset。先前写的是 `frames[skip:] or frames`：
    # 跳完为空时静默退回全部帧（等于不跳），而下面的标签仍然 `+skip` ——
    # 于是五个步号整体多标 60 步，图上写"第 60 步"其实是第 0 步。
    # 图照样出、脚本照样退 0，错的只是纸上那行小字。
    offset = skip if len(frames) > skip else 0
    if offset != skip:
        print(f"⚠ 只渲到 {len(frames)} 帧，不够跳过 {skip} 帧 —— 改为从第 0 帧取，步号跟着改")
    usable = frames[offset:]
    picks = np.linspace(0, len(usable) - 1, KEYFRAME_COUNT).round().astype(int)
    # 每帧先削掉那条纯黑带。**这条不能只在 `crop_to_content` 里做** ——
    # 关键帧这条路根本不裁图（六帧要保持同一取景才比得出动作变化），
    # 于是先前只有开篇拼图干净、讲义里八张关键帧图每格顶上都留着一道黑带。
    strip = [_trim_dark_border(usable[i]) for i in picks]

    use_project_font()
    # 排成 KEYFRAME_COLS 列的网格，**不是一行横排**。
    # 一行横排在屏幕上没问题，进 PDF 就废了：六帧一行的图宽高比 4.4:1，
    # 缩到正文宽（约 16 厘米）之后每帧只有 2.7 厘米，鸭子在纸上小到看不出姿态。
    # 实测第一版 PDF 的取物那张就是这样 —— 图在、看不清，等于没有。
    # 两行三列把宽高比压到 1.3:1，每帧的线性尺寸大 67%、面积近三倍。
    rows = -(-KEYFRAME_COUNT // KEYFRAME_COLS)
    # **每格的高度要按帧的宽高比算，不能写死。** 先前写的是每行 3.4 英寸，
    # 而帧是 4:3、格宽 3.0 英寸 ⇒ 图只占 2.25 英寸高，剩下 1.15 英寸是空白，
    # 十六帧那张在纸上就是四条和图一样高的白带，鸭子却小得看不清姿态。
    # 现在按帧的实际宽高比定行高，再加上步号那行字的高度。
    frame_h, frame_w = strip[0].shape[:2]
    cell_w = 3.6
    fig, axes = plt.subplots(rows, KEYFRAME_COLS,
                             figsize=(cell_w * KEYFRAME_COLS,
                                      (cell_w * frame_h / frame_w + 0.28) * rows), dpi=200)
    flat = np.atleast_1d(axes).ravel()
    for ax, img, idx in zip(flat, strip, picks, strict=False):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"第 {int(idx) + offset} 步", fontsize=9)
    # 帧数不是列数整数倍时，多出来的格子关掉，别留一个空白坐标框。
    for ax in flat[len(strip):]:
        ax.axis("off")
    # 不加 suptitle：任务名在讲义的图题里已经写了，图上再印一遍英文 task id
    # 是给中文读者添一层要忽略的东西。
    fig.tight_layout()
    # 行间还要再收一次。按帧宽高比算完行高之后，每行仍留了一点给步号那行字，
    # imshow 保持等比、图在格子里居中 ⇒ 富余的高度一半跑到图上方去了，
    # 表现为行与行之间一条空带。hspace 收掉它。
    fig.subplots_adjust(hspace=0.02)
    out = result_dir() / f"keyframes-{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"写出 {out}  (取了 {KEYFRAME_COUNT} 帧，共 {len(usable)} 帧可选)")
    return out


def _trained_keyframes(task_dir: Path) -> list[Path]:
    """列出某个任务已渲好的关键帧图，**按迭代号数值排序**，并排除未训练那版。

    这里曾经写的是 `sorted(glob("keyframes-*.png"))[-1]` —— 字典序，两处都错：

    · `keyframes-untrained.png` 排在 `keyframes-iter6000.png` **后面**（`u` > `i`），
      于是"最新"取到的是**随机初始化策略**那张。开篇拼图的标题是
      「这里没有一个动作是人教的」，那一格摆的却是一只根本没学过的鸭子。
      已提交的 cover.png 里走路那格就是这个。
    · `iter400` 排在 `iter2000` 后面（`4` > `2`），轮滑与起身两格取到了 400 而不是 2000。

    同一个文件里 `latest_checkpoint()` 已经在用数值排序（`int(stem.split("_")[1])`）——
    两种写法并存，错的那种没人看出来，因为它**照样返回一个存在的文件**。

    Args:
        task_dir: `result/<task-slug>/` 目录。

    Returns:
        按迭代号升序的路径列表；目录不存在或只有未训练那版时返回空列表。
    """
    if not task_dir.is_dir():
        return []
    numbered = []
    for p in task_dir.glob("keyframes-iter*.png"):
        m = re.fullmatch(r"keyframes-iter(\d+)", p.stem)
        if m:
            numbered.append((int(m.group(1)), p))
    return [p for _n, p in sorted(numbered)]


def _trim_dark_border(img: np.ndarray) -> np.ndarray:
    """削掉帧四周那条纯黑带 —— 渲染缓冲区没画到的部分。

    削多少不能写死：实测同一批帧里这条带子在不同任务上从 0 到 30 像素不等
    （走路那张顶部 30 行是精确的 0.000，旋转那张一行都没有）。写死 6 像素
    等于对一半的图没削干净，而剩下的黑边裁到边缘就成了拼图上一道黑杠。
    所以按亮度削：从边往里，整行/整列均值低于 `_DARK_LEVEL` 就丢掉。

    Args:
        img: (H, W, 3) 的帧。

    Returns:
        削过边的帧；判不出黑边时原样返回。
    """
    a = img.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    g = a.mean(axis=2)
    h, w = g.shape
    cap_h, cap_w = int(h * _DARK_MAX_FRAC), int(w * _DARK_MAX_FRAC)
    top = 0
    while top < cap_h and g[top].mean() < _DARK_LEVEL:
        top += 1
    bot = h
    while bot > h - cap_h and g[bot - 1].mean() < _DARK_LEVEL:
        bot -= 1
    left = 0
    while left < cap_w and g[:, left].mean() < _DARK_LEVEL:
        left += 1
    right = w
    while right > w - cap_w and g[:, right - 1].mean() < _DARK_LEVEL:
        right -= 1
    return img[top:bot, left:right]


def crop_to_content(img: np.ndarray, margin_frac: float = 0.06,
                    aspect: float | None = None) -> np.ndarray:
    """裁掉没有信息的边，并把主体**放到画面正中**。

    两件事都是踩出来的：

    · **判据要按「块」而不是按像素。** 地板上那些白色格线的三通道最小值也很高，
      按像素找包围盒时它们横贯整幅、纵贯整幅 —— 包围盒等于整张图，等于没裁。
      先把亮度图按 `_BLOCK` 做均值池化再判：一条两像素宽的线在 8×8 的块里只占
      四分之一权重，均值掉到阈值以下；鸭子是一整块实心的白，均值仍然高。
    · **裁完要以主体为中心。** 只按包围盒加等比边距，主体在原图哪个位置就还在哪个位置 ——
      实测走路那张的鸭子在左上角，裁完仍在左上角。所以最后一步是：取包围盒中心，
      按目标尺寸对称地往两边扩。

    Args:
        img: (H, W, 3) 的帧。
        margin_frac: 主体包围盒之外留多少（按包围盒边长的比例）。
        aspect: 想要的宽高比；给 None 就沿用原图的宽高比。

    Returns:
        裁过的帧；判不出主体时原样返回（宁可不裁，也不要裁出一张空图）。
    """
    img = _trim_dark_border(img)
    a = img.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    lum = a.min(axis=2)          # 白/灰的机器人三通道都高；饱和色的调试箭头与暗地板都低
    h, w = lum.shape
    bh, bw = h // _BLOCK, w // _BLOCK
    if bh < 4 or bw < 4:
        return img
    pooled = lum[: bh * _BLOCK, : bw * _BLOCK].reshape(bh, _BLOCK, bw, _BLOCK).mean(axis=(1, 3))
    mask = pooled > _CONTENT_LEVEL
    if mask.sum() < 4:
        return img
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    y0, y1 = rows[0] * _BLOCK, (rows[-1] + 1) * _BLOCK
    x0, x1 = cols[0] * _BLOCK, (cols[-1] + 1) * _BLOCK

    # 以主体中心为中心，按目标宽高比取一个尽量大的框
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    want_h = (y1 - y0) * (1 + 2 * margin_frac)
    want_w = (x1 - x0) * (1 + 2 * margin_frac)
    ar = aspect if aspect else w / h
    want_w = max(want_w, want_h * ar)
    want_h = want_w / ar
    # 贴边时**整体缩小**，不能把某一边切齐主体 —— 那就成了"头顶到画面上沿"。
    # 先算四个方向各自还能扩多远，取最紧的那个当半高/半宽。
    half_h = min(want_h / 2, cy, h - cy)
    half_w = min(want_w / 2, cx, w - cx)
    half_h = min(half_h, half_w / ar)
    half_w = half_h * ar
    # 主体到裁框上/下沿至少要留这么多（按主体高度的比例）。留不出来就把框再缩，
    # 缩到主体本身放不下时才放弃 —— 那说明原图机位就太近，该退远重渲。
    need = (y1 - y0) * _MIN_HEADROOM
    if half_h < (y1 - y0) / 2 + need:
        half_h = min((y1 - y0) / 2 + need, cy, h - cy)
        half_w = half_h * ar
        if half_w > min(cx, w - cx):
            half_w = min(cx, w - cx)
            half_h = half_w / ar
    yy0, yy1 = int(round(cy - half_h)), int(round(cy + half_h))
    xx0, xx1 = int(round(cx - half_w)), int(round(cx + half_w))
    if yy1 - yy0 < 32 or xx1 - xx0 < 32:
        return img
    # **裁出来的图不能太小。** 相机退远之后鸭子在原图里占得少，紧贴主体去裁会得到
    # 一张两百来像素的小图 —— 放进拼图再放大就是一团糊。裁框不小于原图高度的
    # `_MIN_KEEP`，宁可多留一点地板，也不要分辨率不够。
    min_h = int(h * _MIN_KEEP)
    if yy1 - yy0 < min_h:
        half_h = min(min_h / 2, cy, h - cy)
        half_w = min(half_h * ar, cx, w - cx)
        half_h = half_w / ar
        yy0, yy1 = int(round(cy - half_h)), int(round(cy + half_h))
        xx0, xx1 = int(round(cx - half_w)), int(round(cx + half_w))
    return img[yy0:yy1, xx0:xx1]


def _latest_parallel(task_dir: Path) -> Path | None:
    """某个任务最新那张并行仿真图，按迭代号数值排序。

    与 `_trained_keyframes` 同一个判据（数值序），只是文件名前缀不同。
    两处都不能用字典序 —— `iter400` 会排在 `iter2000` 后面。

    Args:
        task_dir: `result/<task-slug>/` 目录。

    Returns:
        迭代号最大的那张并行图；没有就返回 None。
    """
    if not task_dir.is_dir():
        return None
    numbered = []
    for p in task_dir.glob("parallel-iter*.png"):
        m = re.fullmatch(r"parallel-iter(\d+)", p.stem)
        if m:
            numbered.append((int(m.group(1)), p))
    return max(numbered)[1] if numbered else None


def render_cover_tile(checkpoint: str | None = None) -> Path:
    """渲一张**单只**鸭子的定格，专供开篇拼图用。

    为什么不复用别的两类图：并行图是六只、缩到拼图的一格里每只只剩几十像素；
    关键帧是一张 2×3 的拼版，再拼进拼图就成了"拼图的拼图"。
    开篇那张要的是"一个动作一眼认得出"，所以每格该是**一只鸭子的一帧**。

    取哪几帧：`_COVER_TILE_FRACS` 里的每个相位各一张，全部写出来当候选。
    只取一帧是碰运气 —— 那一刻可能正好在腾空、在侧身；一个动作在回合里反复出现，
    不同相位差别很大，机器挑不出"典型"，人一眼就能挑。

    Args:
        checkpoint: 权重路径，None 用随机策略（只验证版面）。

    Returns:
        写出的 PNG 路径。
    """
    # 机位可按进程覆盖。**默认那个正面机位对"姿态类"动作是不够的**：
    # 轮滑与蹲姿滑行从正面看几乎一样（都是直立朝镜头），而两者的差别全在腿的弯曲程度上 ——
    # 那要从侧面才看得出来。RL_DUCK_COVER_AZIMUTH 就是为这种情况留的。
    azimuth = os.environ.get("RL_DUCK_COVER_AZIMUTH")
    # 定格的机位比关键帧那档**退远一些**（0.8 → 0.95 米）。
    # 理由是构图不能靠裁来救：跟拍相机对准的是髋部，而头在髋部之上 ——
    # 用 0.8 米拍出来鸭子顶着画面上沿，裁的时候上方根本没有余量可留，
    # 只能把头贴着边切出去。先在原图里留出空间，裁才有得选。
    # 退到 1.2 米又过了头：鸭子在原图里太小，紧贴主体裁出来只有两百来像素 ——
    # 0.95 米配 `crop_to_content` 的 `_MIN_KEEP` 下限，两头都够。
    env = DuckEnv(
        num_envs=1, device="cuda:0", seed=1, render_mode="rgb_array", shadows=False,
        render_size=KEYFRAME_SIZE, camera_distance=_COVER_TILE_DISTANCE,
        camera_elevation=KEYFRAME_ELEVATION, camera_origin="asset_root",
        camera_azimuth=float(azimuth) if azimuth else None,
        debug_vis=False,  # 关掉指令箭头，见 env.py 那段注释
    )
    frames: list[np.ndarray] = []
    try:
        model, tag = make_policy(env, checkpoint)
        skip, steps = keyframe_window(env)
        obs, _critic = env.reset()
        for _ in range(steps):
            with torch.no_grad():
                actions = model.act_inference(obs)
            obs, _critic, _r, _d, _i = env.step(actions)
            rendered = env.render()
            if rendered is not None:
                frames.append(_as_uint8(rendered))
    finally:
        env.close()
    if not frames:
        raise RuntimeError("render() 一帧都没回 —— 检查 render_mode 与 viewer 配置")
    usable = frames[skip:] if len(frames) > skip else frames
    # **多存几张候选，让人挑。** 只取中段那一帧是碰运气：那一刻可能正好在腾空、
    # 在侧身、或者刚被推了一下 —— 而开篇拼图要的是"这个动作最典型的样子"。
    # 一个动作在回合里反复出现，不同相位差别很大，机器挑不出"典型"，人一眼就能挑。
    out = None
    for frac in _COVER_TILE_FRACS:
        pick = usable[min(int(len(usable) * frac), len(usable) - 1)]
        cand = result_dir() / f"cover-cand-{tag}-{int(frac * 100):02d}.png"
        media.write_image(cand, crop_to_content(pick, margin_frac=0.10, aspect=4 / 3))
        out = out or cand
    print(f"写出 {len(_COVER_TILE_FRACS)} 张候选到 {result_dir()}/cover-cand-{tag}-*.png"
          f" —— build_cover 会按相位编号取前 {COVER_COLUMNS} 张排成一行")
    return out


def build_cover() -> Path | None:
    """把每个动作的三个瞬间摆成开篇那条**动作带**：一行一个动作，行内三张按时间排。

    为什么一行三张、而不是一格一张：一格一张只证明"这一刻它站着"。
    走路要看两条腿在换、前滚翻要看从起手到落地 —— 动作是**过程**，
    一张定格拍不出过程，三张连起来一眼就看出来了。

    动作名写在左边的空白里，不烧到图上：图上一有字，排版换了、措辞改了都动不了它，
    而且与正文的图题重复。

    取图顺序按候选的相位编号（`cover-cand-<tag>-<NN>.png` 里的 `NN`），
    所以行内三张天然是时间顺序。

    Returns:
        写出的 PNG 路径；一张都凑不出来时返回 None。

    Raises:
        RuntimeError: `XBOTICS_FIG_FONT` 取不到 —— 回落系统字体会静默换掉字形。
    """
    root = Path(__file__).resolve().parents[1] / "result"
    font_path = os.environ.get("XBOTICS_FIG_FONT", "")
    if not font_path or not Path(font_path).is_file():
        raise RuntimeError(f"出图字体取不到（西文 Times New Roman + 中文宋体）：{font_path!r}")

    rows: list[tuple[str, list[Path]]] = []
    for slug, label in COVER_LAYOUT:
        d = root / slug
        cand = sorted(d.glob("cover-cand-*.png"), key=lambda q: q.stem.split("-")[-1]) if d.is_dir() else []
        if not cand:
            pick = _latest_parallel(d)
            cand = [pick] if pick else []
        if not cand:
            print(f"动作带缺一行：{label}（{slug} 下没有候选定格）")
            continue
        rows.append((label, cand[:COVER_COLUMNS]))
        print(f"动作带取 {label}（{slug}）: {' '.join(q.name for q in cand[:COVER_COLUMNS])}")
    if not rows:
        print("一行都凑不出来，先渲各动作的单只定格。")
        return None

    tw, th = _COVER_TILE_PX
    gut = _COVER_GUTTER_PX
    width = _COVER_LABEL_PX + tw * COVER_COLUMNS
    height = (th + gut) * len(rows) - gut
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(font_path, 40)
    for r, (label, paths) in enumerate(rows):
        y = r * (th + gut)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((_COVER_LABEL_PX - (box[2] - box[0]) - 28, y + (th - (box[3] - box[1])) // 2 - box[1]),
                  label, font=font, fill="black")
        for c, q in enumerate(paths):
            sheet.paste(Image.open(q).convert("RGB").resize((tw, th), Image.LANCZOS),
                        (_COVER_LABEL_PX + c * tw, y))
    out = root / "cover.png"
    sheet.save(out)
    print(f"写出 {out}  ({len(rows)} 行 × {COVER_COLUMNS} 张)")
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

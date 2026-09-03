"""把小鸭子的 mjlab 任务包成课程用的薄接口。

这是整个 code/ 目录里**唯一与机器人有关**的文件。物理、观测、奖励、终止条件全在
`mjlab_microduck` 里（那是 pyproject 钉了 commit 的依赖），这里只做三件事：按 `TASK`
从 mjlab 的任务注册表取出环境配置、把并行环境数定下来、把 mjlab 按组返回的观测拆成
actor 和 critic 各自那一份。三个训练版本共用本文件，对照时环境完全一致。

**换任务只改 `TASK` 那一行。** 注册表里 33 个任务（13 个平地主任务 + 碎石地与带齿隙孪生）
共用同一份训练代码，这正是第 8 节要证明的事。

讲义对应：第 2 节（观测与动作）、4.9 节（代码地图）、5.5 节（代码走读）、8.1 节（换任务）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# 导入即注册：这一行执行 mjlab_microduck 里所有 register_mjlab_task(...) 调用，
# 之后 registry 才认得下面那些 task id。不靠 entry-point 自动发现，是为了让
# "什么时候注册的" 这件事在代码里看得见。
import mjlab_microduck.tasks  # noqa: F401
import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

# ============================ 这一轮跑什么 ============================
# 全仓唯一的任务声明处。改这一行就是换一个动作；结果目录与权重目录都从它派生，
# 所以不同任务的产物不会互相覆盖。可选值用 `python code/env.py` 打印。
_DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"
# 批量跑的时候（一台机器七张卡、每卡一个任务）用环境变量按进程覆盖 ——
# 模块级常量是一份，七个并发进程要各跑不同任务，只能从进程环境里取。
# 讲义里「改一行换任务」说的是单次运行，不受这个影响。
TASK = os.environ.get("RL_DUCK_TASK") or _DEFAULT_TASK

# 并行环境数。三级阶梯的对照要求三个版本用同一个值，所以它声明在这里、
# 不写在各 train 脚本的 main() 里——写三份就会漂。
# 取 4096 还是更多？在一张 48 GB 的卡上实测了一条纯仿真吞吐曲线（150 步，不含学习）：
#   2048 →  15.98 步/秒     4096 → 15.00 步/秒     8192 → 10.70 步/秒     16384 → 5.49 步/秒
# 换成"每墙钟秒采到多少环境步"：0.033 / 0.061 / 0.088 / 0.090 百万。
# ⇒ 2048→4096 几乎免费（说明还被核函数启动延迟卡着），4096→8192 净赚约 1.45 倍，
#   8192→16384 完全打平（吞吐到顶）。**甜点在 8192。**
# 显存从来不是约束：16384 个环境时 torch 峰值才 0.5 GiB，整卡占用不到 10 GB。
_DEFAULT_NUM_ENVS = 4096
NUM_ENVS = int(os.environ.get("RL_DUCK_NUM_ENVS") or _DEFAULT_NUM_ENVS)

# 训练多少个迭代。一次迭代 = NUM_ENVS × 24 步。
# 走路取 6000：这个任务的奖励表里「动作变化太快」那一项的权重由环境的课程表
# 从 -0.1 分段拉到 -1.0、**到第 1500 迭代才到位**，跑 3000 就只有一半时间是在
# 最终那套奖励下学的，三档对照会被这件事污染。上游 playbook 给步态类的量级是
# 4000–6000；episodic 的动作类 1000 上下，换任务时把这里改成 2000 即可。
_DEFAULT_MAX_ITERATIONS = 6000
MAX_ITERATIONS = int(os.environ.get("RL_DUCK_MAX_ITERATIONS") or _DEFAULT_MAX_ITERATIONS)

# 随机种子。三档对照必须同种子，所以它也声明在这里而不是各 main() 里。
# 可覆盖的理由：难的动作（起身）一次运行成不成有随机性，同配方换种子并行开两档、
# 取好的那个，比串行等一档跑完再决定省几个小时。**换了种子就不再是同一次实验**——
# 讲义里引用的曲线一律用默认种子那档。
_DEFAULT_SEED = 1
SEED = int(os.environ.get("RL_DUCK_SEED") or _DEFAULT_SEED)
# =====================================================================


def task_slug(task: str = TASK) -> str:
    """把 task id 压成短名，供 run 名与目录名用。

    名字是从 `TASK` 算出来的，不是随手起的——同一个 task 永远得到同一个目录名。

    Args:
        task: mjlab 的 task id，默认取模块级的 `TASK`。

    Returns:
        小写连字符短名，例如 `Mjlab-Velocity-Flat-MicroDuck` → `velocity-flat`。
    """
    name = task.removeprefix("Mjlab-").replace("-MicroDuck", "")
    return name.strip("-").lower()


def task_symmetry_cfg(task: str = TASK) -> dict | None:
    """取任务自己声明的左右对称性增广配置。

    对称性是**算法侧**的东西（一个镜像一致性损失，加在 PPO 的 loss 上），不像奖励表和
    课程表那样由 mjlab 内部驱动 —— 所以训练器必须自己去问任务要不要它。13 个平地主任务里
    只有前滚翻声明了 True（那个动作是严格左右对称的，镜像一致性正好压住"往一侧塌"）。

    值从 mjlab 的任务注册表里取，不在这边另列一份任务名到开关的表：注册表是唯一声明处，
    抄一份出来就会在上游改了开关那天静默失配。

    `.algorithm` 直接取属性、不用 `getattr(..., None)`：上游改名时要当场 `AttributeError`。
    用默认值兜住它是这里最危险的写法 —— 兜住的结果是**镜像损失整个静默关掉**，
    而 33 个任务里除前滚翻之外全都本来就返回 None，看日志分不出"这个任务不需要"和"字段名找不到了"。

    `symmetry_cfg` 那一层保留 `getattr` 默认值，因为**缺席就是"不启用"是上游自己的语义**：
    只有前滚翻用的 `PpoWithSymmetryCfg` 有这个字段，其余任务用的基类 `RslRlPpoAlgorithmCfg`
    根本没有它。这不是我们的兜底，是在读上游的两种配置类型。

    Args:
        task: mjlab 的 task id，默认取模块级的 `TASK`。

    Returns:
        任务声明的 symmetry 配置字典；任务用的是不带该字段的基类配置时返回 None。

    Raises:
        AttributeError: 上游的 runner 配置不再有 `algorithm` 字段。
    """
    return getattr(load_rl_cfg(task).algorithm, "symmetry_cfg", None)


def split_actor_critic_obs(obs):
    """把 mjlab 按组返回的观测拆成 actor 那份和 critic 那份。

    两份不一样是有意的：critic 只在训练时存在，可以多看仿真器里的"天眼"信息
    （基座真实线速度、未加编码器偏置的真关节角之类），把分打得更准；actor 部署时
    要单独带走，只能看真机上拿得到的。

    Args:
        obs: mjlab 返回的观测字典，键为观测组名。

    Returns:
        (actor 观测, critic 观测) 两个张量。
    """
    actor_obs = obs["actor"] if "actor" in obs else obs["policy"]
    return actor_obs, obs["critic"]


def result_base() -> Path:
    """过程产物的根目录。

    默认是仓内的 `result/`（独立克隆下来的人跑一下就有东西看）；
    本仓库把它指到仓外，理由是 `rl_duck/` 这个目录本身就是交付物 ——
    交付物里不该混着几百兆的中间件。用 `RL_DUCK_RESULT_ROOT` 覆盖。

    Returns:
        过程产物根目录的绝对路径（已建好）。
    """
    override = os.environ.get("RL_DUCK_RESULT_ROOT")
    base = Path(override) if override else Path(__file__).resolve().parents[1] / "result"
    base.mkdir(parents=True, exist_ok=True)
    return base


class DuckEnv:
    """小鸭子的 mjlab 任务环境，三个算法版本共用。

    默认是平地速度指令行走：每个回合给一个随机目标速度（前进 / 侧移 / 转向）＋一个
    头部姿态指令，奖励让它跟上指令并保持直立。换 `TASK` 就换成别的动作，本类不用改——
    actor 侧所有任务同一份布局：**61 维**，动作都是 14 个舵机。
    critic 侧按任务不同（实测取物/前滚翻 74、走路三档 76、旋转/轮滑 78）——
    它只在训练时存在，各任务想让它多看什么由各自的环境配置定。
    两个数不一样是有意的（见 `split_actor_critic_obs`）—— 实测 critic 侧多出的 15 维是
    基座真实线速度 3、双脚高度 2、双脚腾空时间 2、双脚接触 2、接触力 6。
    先前这里只写「同一份 61 维」，那句话漏掉了 critic 那一侧。

    回合长度不在这里覆盖：走路是长回合，起身、坐站那些是短回合，各任务的配置里
    已经按自己的需要定好了，硬塞一个统一值会把 episodic 任务弄坏。
    """

    def __init__(
        self,
        num_envs=NUM_ENVS,
        device="cuda:0",
        seed=None,
        render_mode=None,
        task=TASK,
        max_extra_envs=0,
        render_size=None,
        camera_distance=None,
        camera_elevation=None,
        camera_azimuth=None,
        camera_lookat=None,
        env_spacing=None,
        shadows=None,
        camera_origin=None,
        debug_vis=None,
    ):
        if task not in list_tasks():
            raise ValueError(f"未注册的任务 {task!r}；可选值见 `python code/env.py`")
        # `camera_lookat` 只在自由相机（`camera_origin="world"`）下生效，理由见下面
        # 那段实测注释。签名是扁平的、表达不出参数之间的依赖，所以这条约束只能在这里
        # 机械化 —— 而它已经付过一次代价：出关键帧的那处传了 lookat、没传 origin，
        # 于是 lookat 是死值，而画面看着是对的（跟拍恰好把鸭子放中间），没人会发现。
        # 前提不成立就抛错，不要什么都不发生。
        if camera_lookat is not None and camera_origin != "world":
            raise ValueError(
                "camera_lookat 只在 camera_origin='world' 下生效（跟拍模式下 MuJoCo 每帧用被跟"
                f"刚体的位置覆盖注视点）。当前 camera_origin={camera_origin!r}："
                "要对准一个指定的点就传 camera_origin='world'；要跟着机器人就用 'asset_root' 并去掉 lookat。"
            )
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        # 环境之间的摆放间距。每个环境是一份独立的物理世界，间距只影响画面上的偏移
        # 与地形分片查询 —— 平地任务上把它调小纯属视觉安排，不改物理。
        # 出「几只一起」那张图时把它压小，鸭子才挨得近、拍得大。
        if env_spacing is not None:
            cfg.scene.env_spacing = env_spacing
        # 训练与评测只渲第 0 个环境（省时间）；出「一屏一群鸭子」那张图时把它调大。
        cfg.viewer.max_extra_envs = max_extra_envs
        # mjlab 的默认渲染分辨率是 320×240 —— 够看不够印。要进讲义的图必须显式放大，
        # 否则渲出来的东西照样能存成 PNG、脚本照样退 0，只是印在纸上是一团马赛克。
        if render_size is not None:
            cfg.viewer.width, cfg.viewer.height = render_size
        # 相机默认距离 5 米、看向原点。对一只 25 厘米高的鸭子来说：单只太小、一群又装不下，
        # 而且默认视线几乎水平 —— 地平线落在画面中间，上面三四成是纯黑的天。
        # 三个量一起调才有用：拉近/拉远（distance）、往下看（elevation）、
        # 把视线抬到鸭子躯干高度而不是地面（lookat）。
        if camera_distance is not None:
            cfg.viewer.distance = camera_distance
        if camera_elevation is not None:
            cfg.viewer.elevation = camera_elevation
        # 方位角：几个环境的原点是排成一条线的，默认 90 度会让那条线在画面里斜着走 ——
        # 一头顶出边缘、另一头空一片。让相机从垂直于那条线的方向看，才横铺满画面。
        if camera_azimuth is not None:
            cfg.viewer.azimuth = camera_azimuth
        # ⚠️ `lookat` 只在**自由相机**下生效。mjlab 的默认 origin_type 是 AUTO，
        # 而 AUTO 的语义是「跟拍第一个非固定刚体」—— 跟拍模式下 MuJoCo 每帧用被跟刚体的
        # 位置覆盖注视点，配置里的 lookat 就成了死值。实测证据：把 lookat 平移
        # (1.12, -0.56) 米重渲，两张图逐像素相同。
        # ⇒ 要对准一个**指定的点**（比如一群的中心）必须显式选 "world"；
        #   要跟着一个**会走的机器人**就选 "asset_root"。两者不能混。
        if camera_lookat is not None:
            cfg.viewer.lookat = camera_lookat
        # 阴影与地面反射默认都开着。问题是这个场景是「25 厘米的机器人 + 一望无际的地面」，
        # 阴影贴图的深度范围被地面撑得极大、精度不够，渲出来是一道道横向条纹（阴影失真）。
        # 出图时一律关掉：格子地板本身就够表达空间关系，条纹只会让人以为画面坏了。
        if shadows is not None:
            cfg.viewer.enable_shadows = shadows
            cfg.viewer.enable_reflections = shadows
        # 指令的调试箭头（`twist` 那一项默认 debug_vis=True，画一根蓝色速度箭头，
        # 有时还有一根绿色的竖轴）。它是**调试用的可视化**，不是机器人的一部分 ——
        # 出图时必须关掉，否则读者会以为那是场景里的东西。
        if debug_vis is not None:
            cmds = cfg.commands
            items = cmds.items() if hasattr(cmds, "items") else vars(cmds).items()
            for name, c in list(items):
                if not name.startswith("_") and hasattr(c, "debug_vis"):
                    c.debug_vis = debug_vis
        # 给录视频用的固定指令。**这是必须有的**：指令是每个回合随机采的，
        # 而轮滑那档的范围是 (-0.5, 0.6)、蹲姿滑行是 (-1.0, 1.0) —— 都跨过零。
        # 采到接近零的那次，"站着不动"就是正确行为，于是录出来的八秒里鸭子一动不动，
        # 而这既不是策略坏了也不是渲染坏了，看视频的人只会以为它没学会。
        # 写法：RL_DUCK_FORCE_CMD="lin_vel_x=0.5,lin_vel_y=0,ang_vel_z=0,heading=0"
        forced = os.environ.get("RL_DUCK_FORCE_CMD")
        if forced:
            want = dict(kv.split("=", 1) for kv in forced.split(",") if "=" in kv)
            cmds = cfg.commands
            items = cmds.items() if hasattr(cmds, "items") else vars(cmds).items()
            for name, c in list(items):
                r = getattr(c, "ranges", None) if not name.startswith("_") else None
                if r is None:
                    continue
                for key, value in want.items():
                    # 原值是 None 表示这一档不用这个字段（轮滑的 heading 就是 None），
                    # 给它塞一个区间会改变任务语义 —— 只钉本来就在用的字段。
                    if getattr(r, key, None) is not None:
                        setattr(r, key, (float(value), float(value)))

        if camera_origin is not None:
            origin_types = {
                "world": cfg.viewer.OriginType.WORLD,        # 自由相机，看向 lookat
                "asset_root": cfg.viewer.OriginType.ASSET_ROOT,  # 跟拍机器人根节点
            }
            cfg.viewer.origin_type = origin_types[camera_origin]
            if camera_origin == "asset_root":
                # 场景里不止一个实体（还有地形），跟拍必须点名跟谁。
                cfg.viewer.entity_name = "robot"

        self.task = task
        # 要 GPU 但机器上没有 CUDA ⇒ **报错停下，不回落到 CPU**。
        # 先前这里是 `device if torch.cuda.is_available() or device == "cpu" else "cpu"`，
        # 静默降级。代价不是"慢一点"：MuJoCo Warp 是 GPU 后端，4096 个环境落到 CPU 上
        # 会把一次二十分钟的训练拖成十几小时，而进度条照样在动、日志照样在写 ——
        # 等发现不对时已经烧掉半天。真要跑 CPU 就显式传 device="cpu"。
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                f"要求 device={device!r} 但这台机器上 torch.cuda.is_available() 为假。"
                "本仓不做静默降级 —— MuJoCo Warp 落到 CPU 上会让训练慢两个数量级而不报错。"
                "确实要用 CPU 就显式传 device=\"cpu\"。"
            )
        resolved_device = device
        self._env = ManagerBasedRlEnv(cfg=cfg, device=resolved_device, render_mode=render_mode)
        self.device = torch.device(self._env.device)
        self.num_envs = self._env.num_envs
        self.action_dim = int(self._env.single_action_space.shape[0])
        self.metadata = self._env.metadata

        obs, critic_obs = self.get_observations()
        self.obs_dim = int(obs.shape[1])
        self.critic_obs_dim = int(critic_obs.shape[1])

    @property
    def unwrapped(self) -> Any:
        """露出底层的 mjlab 环境对象，供录像等需要原始接口的场合使用。

        Returns:
            未经包装的 mjlab 环境。
        """
        return self._env

    @property
    def episode_steps(self) -> int:
        """一个回合有多少步。

        各任务不一样（走路 20 秒 = 1000 步，起身 6 秒 = 300 步，前滚翻 5 秒 = 250 步），
        唯一声明处是任务配置里的 `episode_length_s`，所以只能问环境要，不能写一个常量。

        Returns:
            回合长度（控制步数）。
        """
        return int(self._env.max_episode_length)

    @property
    def episode_seconds(self) -> float:
        """一个回合有多少秒，也就是任务配置里的 `episode_length_s`。

        为什么要露出"秒"而不只是"步"：判断一个任务是 episodic（那个动作**就是**整个回合）
        还是周期性的（走路，回合只是个时长上限），是关于**任务性质**的判断，用秒表达才读得懂。
        用步数去判会被"一步等于多少秒"这件事拐一道弯 —— 实测踩过：拿出图窗口的 260 步
        当阈值，起身的 300 步回合就被判成了"长回合"，于是那个只有 0.3 秒的起身动作
        照旧落在取帧窗口之外。阈值和绘图长度是两件事，混用一个数就会这样。

        Returns:
            回合时长（秒）。
        """
        return float(self._env.max_episode_length_s)

    def set_curriculum_step(self, step: int) -> None:
        """把课程表的进度计数器搬到指定的环境步，用于评测时对齐训练时的那一档。

        **这是评测口径的一个真坑。** 课程表（初始姿态混合、域随机化幅度、各惩罚项的权重）
        全都按 `env.common_step_counter` 分段，而这个计数器在新建的环境里从 0 开始。
        于是不做对齐时，无论评测的是第几万迭代的权重，环境给的都是**第 0 档**：
        起身那档在第 0 档根本不出现"仰躺"这种初始姿态（它的概率是 0，要到第 600 迭代
        才开始出现），惩罚项也还是最轻的一档。结果就是评测**系统性地把任务测简单了**、
        奖励数字也和训练日志不可比 —— 而它不报错，图照样出、json 照样写。

        上游的 runner 把这个计数器存进 checkpoint、续训时再恢复（`mjlab/rl/runner.py`
        里的 `env_state`），正是为了这件事。我们不存它，因为它可以从已有字段算出来：
        每次迭代恰好推进 `num_steps_per_env` 步，所以第 N 档权重对应 N × num_steps_per_env。

        Args:
            step: 要对齐到的环境步数，通常是 `迭代数 × num_steps_per_env`。
        """
        self._env.common_step_counter = int(step)

    def reset(self):
        """重置全部并行环境，开始新一轮回合。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs, _ = self._env.reset()
        return split_actor_critic_obs(obs)

    def get_observations(self):
        """取当前时刻的观测，不推进仿真。

        采样循环开头要先拿到"这一步看到什么"才能出动作，所以需要一个只读的入口。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs = self._env.observation_manager.compute()
        return split_actor_critic_obs(obs)

    def step(self, actions):
        """把动作送进仿真，推进一个控制周期（50 赫兹，也就是 20 毫秒）。

        mjlab 把"摔倒终止"和"超时截断"分成两个信号返回，这里合并成一个 done 交给
        采样循环——两者对回报截断的作用是一样的。

        Args:
            actions: 形状 (N, 14) 的关节目标量。

        Returns:
            (actor 观测, critic 观测, 奖励, done, 附加信息)。附加信息里**两个原始信号
            都保留**：`time_outs`（PPO 版本用它给超时的回合补一段自举回报）与
            `terminated`（评测要用它把"摔了"和"时间到了"分开 —— 合并后的 `dones`
            分不出这两件事，而短回合的 episodic 任务里超时是常态）。
        """
        obs, rewards, terminated, time_outs, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, time_outs)
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {
            "time_outs": time_outs,
            "terminated": terminated,
            "mjlab_extras": extras,
        }

    def render(self):
        """渲染一帧画面，录 rollout 视频与出关键帧序列时用。

        Returns:
            一帧 RGB 图像；未开启渲染时为 None。
        """
        return self._env.render()

    def close(self):
        """关闭仿真、释放显存。"""
        self._env.close()


def main():
    """打印当前选的任务和全部可选任务，方便改 `TASK` 之前先看一眼。"""
    print(f"当前 TASK = {TASK}  (slug: {task_slug()})，并行环境数 = {NUM_ENVS}")
    print(f"已注册 {len(list_tasks())} 个任务：")
    for name in list_tasks():
        print("  ", name)


if __name__ == "__main__":
    main()

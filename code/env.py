"""把小鸭子的 mjlab 任务包成课程用的薄接口。

这是整个 code/ 目录里**唯一与机器人有关**的文件。物理、观测、奖励、终止条件全在
`mjlab_microduck` 里（那是 pyproject 钉了 commit 的依赖），这里只做三件事：按 `TASK`
从 mjlab 的任务注册表取出环境配置、把并行环境数定下来、把 mjlab 按组返回的观测拆成
actor 和 critic 各自那一份。三个训练版本共用本文件，对照时环境完全一致。

**换任务只改 `TASK` 那一行。** 十几个任务共用同一份训练代码，这正是第 8 节要证明的事。

讲义对应：第 2 节（观测与动作）、4.9 节（代码地图）、5.5 节（代码走读）、8.1 节（换任务）。
"""
from __future__ import annotations

from typing import Any

import torch

# 导入即注册：这一行执行 mjlab_microduck 里所有 register_mjlab_task(...) 调用，
# 之后 registry 才认得下面那些 task id。不靠 entry-point 自动发现，是为了让
# "什么时候注册的" 这件事在代码里看得见。
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

# ============================ 这一轮跑什么 ============================
# 全仓唯一的任务声明处。改这一行就是换一个动作；结果目录与权重目录都从它派生，
# 所以不同任务的产物不会互相覆盖。可选值用 `python code/env.py` 打印。
TASK = "Mjlab-Velocity-Flat-MicroDuck"

# 并行环境数。三级阶梯的对照要求三个版本用同一个值，所以它声明在这里、
# 不写在各 train 脚本的 main() 里——写三份就会漂。
# 上限由显卡显存决定：2080 Ti（11 GB）实测能吃多少就填多少。
NUM_ENVS = 4096
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


class DuckEnv:
    """小鸭子的 mjlab 任务环境，三个算法版本共用。

    默认是平地速度指令行走：每个回合给一个随机目标速度（前进 / 侧移 / 转向）＋一个
    头部姿态指令，奖励让它跟上指令并保持直立。换 `TASK` 就换成别的动作，本类不用改——
    每个任务的观测布局都是同一份 61 维，动作都是 14 个舵机。

    回合长度不在这里覆盖：走路是长回合，起身、坐站那些是短回合，各任务的配置里
    已经按自己的需要定好了，硬塞一个统一值会把 episodic 任务弄坏。
    """

    def __init__(  # noqa: PLR0913 —— 这些是环境的实际自由度，硬合并成一个 cfg 只会多一层
        self,
        num_envs=NUM_ENVS,
        device="cuda:0",
        seed=None,
        render_mode=None,
        task=TASK,
        max_extra_envs=0,
    ):
        if task not in list_tasks():
            raise ValueError(f"未注册的任务 {task!r}；可选值见 `python code/env.py`")
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        # 训练与评测只渲第 0 个环境（省时间）；出「一屏一群鸭子」那张图时把它调大。
        cfg.viewer.max_extra_envs = max_extra_envs

        self.task = task
        resolved_device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
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
            (actor 观测, critic 观测, 奖励, done, 附加信息)。附加信息里保留了
            未合并的 `time_outs`，PPO 版本要用它给超时的回合补一段自举回报。
        """
        obs, rewards, terminated, time_outs, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, time_outs)
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {"time_outs": time_outs, "mjlab_extras": extras}

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

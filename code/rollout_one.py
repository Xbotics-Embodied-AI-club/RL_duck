"""评测并录像**一份指定的权重** —— 交付视频的产出入口。

`rollout.py` 的 `main()` 是"三档同台"那条路：它自己去找 v1/v2/v3 各自最新的权重，
一次跑三档。而交付视频要的是"这一份权重、这个机位、这个指令、这个时长"，
逐个动作各不相同（机位与指令为什么按动作分，见 `render_delivery.py` 的表）。

全部参数走环境变量，因为调用方是 `render_delivery.py` 那张表，不是人手敲：

    RL_DUCK_TASK        任务 id
    RL_DUCK_CHECKPOINT  权重路径
    RL_DUCK_RUN_NAME    产物名（写出 <name>.mp4 与 <name>.json）
    RL_DUCK_VIDEO_STEPS 跑多少步（默认 400）
    RL_DUCK_VIDEO_FIXED 给了就用世界固定机位（默认跟拍）
    RL_DUCK_FORCE_CMD   钉住速度指令，免得采到接近零的指令、录出来像没动

跑法：
    RL_DUCK_TASK=... RL_DUCK_CHECKPOINT=... RL_DUCK_RUN_NAME=... \
      MUJOCO_GL=egl uv run python code/rollout_one.py
"""
from __future__ import annotations

import os

from rollout import run_rollout


def main() -> None:
    """按环境变量评测一份权重并录像。

    Raises:
        SystemExit: 缺 `RL_DUCK_CHECKPOINT` 或 `RL_DUCK_RUN_NAME` —— 不给默认值，
            猜错会把别档的视频覆盖掉。
    """
    ckpt = os.environ.get("RL_DUCK_CHECKPOINT", "")
    name = os.environ.get("RL_DUCK_RUN_NAME", "")
    if not ckpt or not name:
        raise SystemExit("★ 要给 RL_DUCK_CHECKPOINT 与 RL_DUCK_RUN_NAME")
    # 交付视频一律单只：跟拍相机只跟第一只，多开的那几只会在画面里乱入。
    run_rollout(ckpt, name, num_envs=1,
                num_steps=int(os.environ.get("RL_DUCK_VIDEO_STEPS") or 400))


if __name__ == "__main__":
    main()

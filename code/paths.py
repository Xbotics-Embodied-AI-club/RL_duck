"""过程产物落在哪 —— 不依赖仿真器的那一小块路径逻辑。

单独一个文件是为了让**收件**能在没装仿真器的机器上跑。先前 `result_base()` 住在
`env.py` 里，而 `env.py` 顶部 `import mjlab_microduck.tasks` 去注册任务 ——
于是 `collect_delivery.py` 这个纯拷文件的步骤也要求整套 GPU 环境装好才跑得起来。
出图机与收件机不是同一台时，这条耦合会把收件卡死。
"""
from __future__ import annotations

import os
from pathlib import Path


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

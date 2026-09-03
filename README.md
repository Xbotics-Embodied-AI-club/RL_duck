# RL_duck · 小鸭子强化学习入门

一只 800 克、25 厘米高的双足小鸭子，14 个舵机、50 赫兹控制环。这个仓用它讲一条完整的
入门链：**REINFORCE → A2C → PPO**，三个算法在同一个任务上跑给你看；然后同一套 PPO
一行不改，换几个奖励，让它学会滑轮滑、原地旋转、用嘴尖取物、向前翻一个跟头。

讲义在 `docs/`，代码在 `code/`，交付件（讲义 PDF · 图 · 视频 · 权重）在 `交付/`。
出图与录像的过程产物落在 `RL_DUCK_RESULT_ROOT` 指的目录（默认仓内 `result/`）。

## 三条命令

```bash
uv sync                                     # 装环境（独立于其他项目）
uv run pytest tests/ -q                     # 冒烟：环境构得起来、维度对得上
uv run python code/train_v3_ppo.py          # 训一个策略
```

## 换任务只改一行

`code/env.py` 顶部：

```python
_DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"   # 换成下面任意一个 task id
_DEFAULT_NUM_ENVS = 4096                          # 并行环境数；实测甜点是 8192
```

批量跑（一台机器多个任务）时用环境变量按进程覆盖，不必改文件：
`RL_DUCK_TASK` / `RL_DUCK_NUM_ENVS` / `RL_DUCK_MAX_ITERATIONS` / `RL_DUCK_CHECKPOINT`。

可选任务用 `uv run python code/env.py` 打印（上游注册了 33 个：13 个平地主任务，
外加碎石地与带齿隙的孪生版）。
结果目录与权重目录都从 `TASK` 派生，所以换任务不会覆盖上一个任务的产物。
**同一个任务的多次训练要靠文件名区分** —— 三个算法都训 velocity-flat，
所以关键帧、并行图、候选定格的文件名里都带训练名与取样密度（见 `run_label()`）。

## 目录

| 路径 | 里面是什么 |
|---|---|
| `code/env.py` | **唯一与机器人有关的文件**：按 `TASK` 取环境配置，拆 actor / critic 观测 |
| `code/model.py` | `ActorCritic`（高斯 actor + critic + 在线观测归一化）与 GAE 工具，三个版本共用 |
| `code/train_v1_reinforce.py` | REINFORCE：策略梯度 + 常数基线 |
| `code/train_v2_a2c.py` | A2C：加 critic 基线 + GAE |
| `code/train_v3_ppo.py` | PPO：加 clip + 多轮 minibatch 复用 + 按 KL 自适应学习率 |
| `code/rollout.py` | 统一口径评测（确定性动作、16 环境 × 400 步）+ 录像 |
| `code/render_gallery.py` | 并行仿真图、单动作关键帧序列、开篇拼图 |
| `code/plot_reward_curves.py` | 三级阶梯的对照曲线 |
| `<RL_DUCK_RESULT_ROOT>/<task-slug>/` | 每个任务的 json / 曲线 / 关键帧 |
| `code/render_delivery.py` | 交付件的重出入口：一个动作一行，写清权重与机位 |
| `code/collect_delivery.py` | 交付清单：声明交付件是什么，按声明收件 |

三份训练脚本都是同一个 Lightning 结构：在线采样的 `IterableDataset` →
`LightningDataModule` → 算 loss 的 `LightningModule` → `trainer.fit(model, data)`。
**版本之间的差异就是知识点**，横向对照着读。没有命令行参数层 —— 要调的东西写在
第一次用到它的地方。

## 环境与外部依赖

环境是这个仓自己的一套（`~/.venv/rl_duck`），刻意与其他项目分开：这里钉
mjlab 1.3.0 + torch 2.9.1，装不进别处那套 mjlab 1.4 + torch 2.8.0 的环境。

机器人模型、任务配置、BAM 执行器模型、域随机化都来自
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl)（Apache-2.0），
在 `pyproject.toml` 里**钉了具体 commit**装进来。**训练器是我们自己写的** ——
不走上游那套 rsl_rl，因为三份脚本的 diff 才是这份讲义要讲的东西。

权重落 `RL_DUCK_CKPT_ROOT`（默认 `$DATASETS_ROOT/models/trained/rl_duck/`），不进 git。
训练记录走 W&B 离线模式，事后 `wandb sync`。
出图字体从 `XBOTICS_FIG_FONT` 取，取不到就报错停下 —— 回落系统字体会让图看着渲出来了
但字体不对，而且不报错。

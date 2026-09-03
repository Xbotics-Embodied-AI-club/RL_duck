# 第 14 讲 · 强化学习入门（小鸭子版）· 交付件

这个目录放两件：视频与权重。过程产物（各档权重的图、探针留下的中间件）不在这里，
在仓外的 `paper_writing/EAI-wri-025-lecture14-microduck/result/`。

```
交付/
├── 视频/  十三段 mp4
└── 权重/  七份最终权重 + MANIFEST.json（见 权重/README.md）
```

讲义与它引用的十二张图不在这里 —— 讲义源文件在 `docs/`、PDF 在 `docs/pdf/`、
图在 `docs/assets/figures/`。一份东西只放一个位置。

**视频不进 git**（二十多兆，按本仓大文件规矩），文件就在 `交付/视频/`；
克隆一份新的仓不会带上它们，按下面的办法重出即可。图与 PDF 体积小，照常提交。

交付件清单的唯一声明处是 `code/collect_delivery.py` 里的 `FIGURES` 与 `VIDEOS`，
换权重档改那张表，正文一个字都不用动；声明了却不存在的源文件会让它退非 0，不静默跳过。

## 视频

**每个动作两版。** 单只看得清姿态，六只看得出这是同一套策略在一群身上同时跑。
都关掉了速度指令的调试箭头。每一段都抽帧看过 —— 判据是画面，不是 `ffprobe` 那几列数字。

| 单只 | 六只 | 权重 | 画面上该看到什么 |
| --- | --- | --- | --- |
| `单只-走路-v3PPO.mp4` | `六只-走路.mp4` | velocity-flat-ppo · 6000 | 两脚交替离地往前迈 |
| `单只-走路-v1REINFORCE.mp4` | — | velocity-flat-reinforce · 6000 | 两脚基本踩在身体正下方，动的是头 |
| `单只-走路-v2A2C.mp4` | — | velocity-flat-a2c · 6000 | 前倾到几乎跪下之后，换成两腿伸直站宽不动 |
| `单只-轮滑.mp4` | `六只-轮滑.mp4` | velocity-flat-rollers · 2000 | 踩在四个被动轮上**横穿画面** |
| `单只-旋转.mp4` | `六只-旋转.mp4` | spin-flat · 2000 | 原地转，逐帧看头的朝向在变 |
| `单只-取物.mp4` | `六只-取物.mp4` | groundpick-flat · 2000 | 低头把嘴尖压到地面那个点，再起身 |
| `单只-前滚翻.mp4` | `六只-前滚翻.mp4` | roulade-flat · 6000 | 开头一秒多翻过去，之后一直站着 |

`进化-走路-八档迭代.mp4`：随机策略 → 10 → 90 → 170 → 240 → 320 → 400 → 6000 迭代，
每档三秒接成一段共 24 秒，左上角标着训练了多少次。
前七档来自一次**只训 400 迭代、每 10 迭代存一档**的短训练 —— 正式训练每 200 迭代才存一档，
而"学会站、学会迈第一步"全发生在第 200 之前，那一段没有权重可用。
末档接的是正式训练的 6000 迭代。
机位是世界固定的：跟拍会把位移藏起来，而这一段要看的就是"越练越稳"。

**轮滑那段为什么用固定机位**：其余各段是跟拍。轮滑实测 5 秒走 1.553 米，
但跟拍时镜头锁死在鸭子身上、地板又是无特征的格子，看着像原地踏步。
固定机位让它横穿画面，位移才看得见。

## 怎么重出

```sh
# ① 七个动作的图与视频：一个动作一行的配方在 code/render_delivery.py 的 JOBS 表里
RL_DUCK_CKPT_ROOT=交付/权重 RL_DUCK_RESULT_ROOT=<过程产物根> \
  uv run python code/render_delivery.py            # 不给动作名就全做
# ② 三条学习曲线
RL_DUCK_CKPT_ROOT=<训练权重根> RL_DUCK_RESULT_ROOT=<过程产物根> \
  uv run python code/plot_reward_curves.py
# ③ 进化视频
RL_DUCK_EVO_RUN=velocity-flat-ppo RL_DUCK_EVO_FINAL=<6000 迭代那份> \
  uv run python code/render_evolution.py
# ④ 按清单收件（缺一件就退非零）
RL_DUCK_RESULT_ROOT=<过程产物根> uv run python code/collect_delivery.py
# PDF：编译机是 w4（w1 崩过一次、容器重建之后 quarto 与 xelatex 都没了）
paper_writing/EAI-wri-025-lecture14-microduck/render-duck-pdf.sh
```

验收用 `paper_writing/EAI-wri-025-lecture14-microduck/review/make-video-sheets.py`
把每段抽四帧拼成联系表，自己看一遍。这一轮就是靠它抓到了 0.9 秒 45 帧的残片
和 320×240 的低清版 —— 那两个文件的时长与分辨率读数都"不报错"。

# 权重

讲义里每个动作各一份最终权重，七份。`MANIFEST.json` 是逐份实测出来的清单
（迭代号、观测/动作维度、字节数、sha256 前 16 位），不是手写的。

| 目录 | 迭代 | 讲义里是哪一段 |
| --- | --- | --- |
| `velocity-flat-reinforce/` | 6000 | 第 4 节 v1 REINFORCE |
| `velocity-flat-a2c/` | 6000 | 第 5 节 v2 A2C |
| `velocity-flat-ppo/` | 6000 | 第 6 节 v3 PPO（走路） |
| `velocity-flat-rollers-ppo/` | 2000 | 7.2 轮滑 |
| `spin-flat-ppo/` | 2000 | 7.3 原地旋转 |
| `groundpick-flat-ppo/` | 2000 | 7.4 嘴尖取物 |
| `roulade-flat-ppo/` | 6000 | 7.5 前滚翻 |

## 怎么加载

网络的三个维度不写死在代码里，从权重自带的 `training_settings` 读：

```python
import torch
from model import ActorCritic

data = torch.load("velocity-flat-ppo/model_6000.pt", map_location="cpu", weights_only=False)
s = data["training_settings"]
model = ActorCritic(obs_dim=s["obs_dim"], critic_obs_dim=s["critic_obs_dim"],
                    action_dim=s["action_dim"])
model.load_state_dict(data["model"])
model.eval()
```

`code/rollout.py` 的 `load_policy()` 就是这么做的，评测与出图都走它。

## 三件用它之前要知道的

**一、actor 61 维是通用的，critic 不是。** 七份里 actor 一律 61 维、动作一律 14 维；
critic 侧按任务不同（清单里实测：取物与前滚翻 74、走路三档 76、旋转与轮滑 78）。
critic 只在训练时存在，各任务想让它多看什么由各自的环境配置定。

**二、评测要把课程表对齐到这份权重训到的那一档。** 奖励表里有几项的权重随训练推进而变，
新建的环境从第 0 档开始 —— 不对齐会系统性地把任务测简单，而且不报错。
算法在 `rollout.py` 的 `align_curriculum()`：档位 = 迭代号 × 每轮环境步数，
两个数都在 `training_settings` 里。

**三、六份权重里没有 `task` 字段。** 该字段是后加的，只有前滚翻那份带。
`rollout.check_task_match()` 读不到它时会打印一行提示并继续 ——
所以换任务跑之前自己确认权重与 `RL_DUCK_TASK` 对得上：三十多个任务的
观测与动作维度大量重合，错配不会报形状错，只会安静地渲出一段“这个动作没学会”的画面。

## 不进 git

`.pt` 按本仓的大文件规矩不入库（`.gitignore` 里 `*.pt`），文件就放在这个目录里；
`MANIFEST.json` 与本文进 git，清单里的 sha256 可以用来核对拿到的文件对不对。

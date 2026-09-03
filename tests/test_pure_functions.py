"""承重纯函数的检查：不需要 GPU、不需要权重，秒级跑完。

为什么单独一份：`test_env_smoke.py` 需要 CUDA（MuJoCo Warp 只跑在 GPU 上），
所以在开发机上它整份跳过。而下面这些函数是**讲义结论的算术依据** ——
优势怎么算、回报怎么截断、曲线读的是哪一段、拼图取哪张图 —— 它们错了，
出来的图和数看着都正常。这些函数一行 GPU 代码都没有，理应在任何机器上都能验。

覆盖的六个都是踩过或差点踩过的地方：

- `compute_gae` / `reward_to_go`：回合边界要截断。不截断就是把下一个回合的
  回报算进上一个回合，而训练照样收敛，只是收敛到一个错东西上。
- `read_metrics`：`metrics.jsonl` 是追加写的，重跑会让文件里躺着多段。
  实测 velocity-flat-reinforce 那份是 3153 行旧段 + 新段。
- `_trained_keyframes`：先前用字典序取"最新"，`untrained` 排在 `iter6000`
  后面、`iter400` 排在 `iter2000` 后面 —— 开篇拼图里那格摆的是**没训练过**的鸭子。
- `smooth`：长度必须与输入一致，否则画曲线时 x 与 y 对不齐。
- `adapt_learning_rate_to_kl`：夹在上下限之间，且方向不能反。

跑法（在仓根，任何机器）：
    uv run pytest tests/test_pure_functions.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from model import compute_gae
from plot_reward_curves import read_metrics, smooth
from render_gallery import _trained_keyframes
from train_v1_reinforce import reward_to_go
from train_v3_ppo import adapt_learning_rate_to_kl

GAMMA = 0.99
LAM = 0.95


# ─────────────────────────── 回报与优势的回合边界 ───────────────────────────


def test_reward_to_go_discounts_within_an_episode():
    """一个回合内，回报是从后往前的折扣累加。"""
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.zeros(3, 1)
    got = reward_to_go(rewards, dones, GAMMA)
    expected = [1 + GAMMA + GAMMA**2, 1 + GAMMA, 1.0]
    assert got.squeeze(-1).tolist() == pytest.approx(expected, abs=1e-6)


def test_reward_to_go_truncates_at_episode_end():
    """回合在中间结束时，后一段的回报不许渗进前一段。

    第 1 步 done=1 ⇒ 第 0 步只能拿到自己那 1.0；把第 2 步的奖励算进去
    就是把两个回合当成一个，而训练照样跑得动。
    """
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    got = reward_to_go(rewards, dones, GAMMA).squeeze(-1).tolist()
    assert got[1] == pytest.approx(1.0, abs=1e-6)   # 回合最后一步
    assert got[0] == pytest.approx(1.0 + GAMMA * 1.0, abs=1e-6)


def test_compute_gae_matches_hand_computed_two_step():
    """两步、无回合结束，逐项手算对齐。

    δ_t = r_t + γ·V_{t+1} − V_t，A_1 = δ_1，A_0 = δ_0 + γλ·A_1，returns = A + V。
    """
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.zeros(2, 1)
    values = torch.tensor([[0.5], [0.25]])
    next_value = torch.tensor([[0.75]])
    adv, ret = compute_gae(rewards, dones, values, next_value, GAMMA, LAM)

    d1 = 2.0 + GAMMA * 0.75 - 0.25
    d0 = 1.0 + GAMMA * 0.25 - 0.5
    a1 = d1
    a0 = d0 + GAMMA * LAM * a1
    # 两个返回值形状**不一样**：advantages 是 (T, N)（函数里 squeeze 过），
    # returns 是 (T, N, 1)（critic 要回归的目标，保留最后那一维）。
    # 第一版这里对 returns 也只 squeeze 一次，得到嵌套 list 而比不上 —— 拿 flatten 更稳。
    assert adv.flatten().tolist() == pytest.approx([a0, a1], abs=1e-6)
    assert ret.flatten().tolist() == pytest.approx([a0 + 0.5, a1 + 0.25], abs=1e-6)


def test_compute_gae_cuts_the_bootstrap_at_done():
    """回合结束那一步不许把下一个回合的价值自举进来。

    done=1 时 δ 应当只剩 r − V；若截断失效，δ 会多出 γ·V_{t+1} 一项。
    """
    rewards = torch.tensor([[1.0], [0.0]])
    dones = torch.tensor([[1.0], [0.0]])
    values = torch.tensor([[0.5], [10.0]])   # 下一步价值故意给得很大，截断失效会立刻看出来
    next_value = torch.zeros(1, 1)
    adv, _ret = compute_gae(rewards, dones, values, next_value, GAMMA, LAM)
    assert adv.squeeze(-1)[0].item() == pytest.approx(1.0 - 0.5, abs=1e-6)


# ─────────────────────────── 曲线读的是哪一段 ───────────────────────────


def _write_jsonl(path: Path, iterations, rewards):
    path.write_text(
        "".join(json.dumps({"iteration": i, "reward": r}) + "\n"
                for i, r in zip(iterations, rewards, strict=True)),
        encoding="utf-8",
    )


def test_read_metrics_keeps_only_the_last_run(tmp_path):
    """追加写留下的多段里，只取最后一次运行那一段。

    旧段 1–3、新段 1–2。若整份按迭代号排序，两段会交错成一条上下横跳的锯齿 ——
    看着像"训练不稳定"，其实是两次运行叠在了一起。
    """
    _write_jsonl(tmp_path / "metrics.jsonl", [1, 2, 3, 1, 2], [0.1, 0.2, 0.3, 0.7, 0.8])
    it, rw = read_metrics(tmp_path)
    assert it.tolist() == [1, 2]
    assert rw.tolist() == pytest.approx([0.7, 0.8])


def test_read_metrics_single_run_is_untouched(tmp_path):
    """只有一段时，整段都要保留。"""
    _write_jsonl(tmp_path / "metrics.jsonl", [1, 2, 3], [0.1, 0.2, 0.3])
    it, _rw = read_metrics(tmp_path)
    assert it.tolist() == [1, 2, 3]


def test_read_metrics_missing_file_returns_empty(tmp_path):
    """文件不存在时给空数组，让调用方自己决定怎么报 —— 不在这里抛。"""
    it, rw = read_metrics(tmp_path)
    assert it.size == 0
    assert rw.size == 0


def test_smooth_preserves_length():
    """平滑之后长度必须不变，否则画图时 x 与 y 对不齐。"""
    for n in (1, 2, 3, 5, 40):
        v = np.arange(n, dtype=float)
        assert smooth(v).size == n


# ─────────────────────────── 拼图取哪一张关键帧 ───────────────────────────


def test_trained_keyframes_sorts_numerically_and_drops_untrained(tmp_path):
    """按迭代号数值排序，并且排除随机策略那版。

    这两条是同一个真事故的两半：字典序下 `untrained` > `iter6000`（`u` > `i`）、
    `iter400` > `iter2000`（`4` > `2`）。已提交的开篇拼图里，走路那格取到的
    就是 `keyframes-untrained.png`，而图标题写着「这里没有一个动作是人教的」。
    """
    for name in ("keyframes-iter400.png", "keyframes-iter2000.png",
                 "keyframes-iter6000.png", "keyframes-untrained.png"):
        (tmp_path / name).touch()
    got = [p.name for p in _trained_keyframes(tmp_path)]
    assert got == ["keyframes-iter400.png", "keyframes-iter2000.png", "keyframes-iter6000.png"]
    assert got[-1] == "keyframes-iter6000.png"      # "最新" 取的是这个


def test_trained_keyframes_only_untrained_gives_nothing(tmp_path):
    """只有随机策略那版时返回空 —— 拼图应当缺一格，而不是摆一只没学过的鸭子。"""
    (tmp_path / "keyframes-untrained.png").touch()
    assert _trained_keyframes(tmp_path) == []


def test_trained_keyframes_missing_dir_gives_nothing(tmp_path):
    """目录不存在时返回空，不抛 —— 还没训的任务本来就没有这个目录。"""
    assert _trained_keyframes(tmp_path / "nope") == []


# ─────────────────────────── KL 自适应学习率 ───────────────────────────


def _lr_after(kl, start_lr, desired_kl=0.01):
    """跑一次学习率调整，给出调整后的值。

    `kl` 必须是**张量**：那个函数里写的是 `kl.detach().cpu()`（训练时传进去的是
    张量）。这份测试第一版传了 float，于是 AttributeError —— 测试自己把签名试出来了。
    """
    opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=start_lr)
    adapt_learning_rate_to_kl(opt, torch.tensor(float(kl)), desired_kl)
    return opt.param_groups[0]["lr"]


def test_lr_shrinks_when_kl_too_large():
    """KL 超标 ⇒ 减速。方向反了会让训练在走猛之后继续加速。"""
    assert _lr_after(0.05, 1e-3) < 1e-3


def test_lr_grows_when_kl_too_small():
    """KL 偏小 ⇒ 加速。"""
    assert _lr_after(0.001, 1e-3) > 1e-3


def test_lr_stays_within_bounds():
    """反复触发同一侧也不许越过上下限。"""
    lr = 1e-3
    for _ in range(50):
        lr = _lr_after(0.5, lr)          # 一路超标
    assert lr == pytest.approx(1e-5, rel=1e-6)
    for _ in range(50):
        lr = _lr_after(1e-6, lr)         # 一路偏小（**严格大于零**，见下一个测试）
    assert lr == pytest.approx(1e-2, rel=1e-6)


def test_lr_unchanged_when_kl_is_exactly_zero():
    """KL 恰好为 0 时不动学习率 —— 那是"这次更新什么都没发生"，不是"步子太小"。

    条件写的是 `0.0 < kl < 0.5 * desired_kl`，下界是严格的。这份测试第一版
    拿 kl=0.0 去测"加速"，于是学习率一动不动、断言失败 —— 把这个边界
    钉下来，以后谁把下界改成 `<=` 会当场红。
    """
    assert _lr_after(0.0, 1e-3) == pytest.approx(1e-3, rel=1e-9)

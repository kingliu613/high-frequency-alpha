# 因子库中文导读

> 目标读者：不太懂代码、但要理解当前高频因子库逻辑的人。
> 当前默认因子库只保留有清晰公式、字段和过程依据的方向因子。

---

## 一、整体逻辑

```
数据快照 -> 因子 -> 加权合成 -> 回测 -> IC/夏普评估
```

把市场想成两条排队的人龙：买方队伍 bid 和卖方队伍 ask。
每 3 秒拍一张照片，记录两边各 10 个价位排了多少量，以及刚才谁成交了。
因子就是用这些照片和成交记录猜下一步价格方向的一种读法。

---

## 二、文件地图

| 文件 | 干什么 |
|---|---|
| `research/hft_factors_china.md` | 当前因子目录 + 文献依据 |
| `src/data/synthetic.py` | 造模拟行情,用于测试代码 |
| `src/data/loader.py` | 接真实数据 |
| `src/signals/ofi.py` | OFI 与成交失衡 |
| `src/signals/features.py` | 队列失衡、涨跌停状态 |
| `src/signals/auction.py` | 开盘集合竞价信号 |
| `src/signals/advanced.py` | API、OEI、羊群、撤单、风险闸门 |
| `src/signals/composite.py` | 因子权重、分组、合成 |
| `src/signals/daily.py` | 日频聚合和截面评估 |
| `src/backtest/engine.py` | T+1、费用、涨跌停约束 |

---

## 三、默认方向因子

### 看订单流

- `mlofi`：多档订单流失衡，看盘口变化的净方向。
- `agg_ofi`：一段时间内累计的 OFI，看持续压力。
- `trade_imbalance`：按论文公式计算买方人数/订单数 vs 卖方人数/订单数的 polarity。

### 看成交/挂单结构

- `api`：显式订单事件里的吃单量相对挂单供给。
- `oei`：按 Chi 论文的 `L-C-M` 事件公式比较买卖两侧。

## 四、诊断项和标签，不进默认合成

- `queue_imbalance`：最优买一队列 vs 卖一队列谁更厚，只作为盘口诊断。
- `herding`：单票 LSV 风格读数只作诊断；严格复现需要截面或投资者群体数据。
- `cancel_spike`：单边撤单爆发只作诊断；还不是 spoofing 论文里的完整检测器。
- `auction_signal`：开盘集合竞价失衡只保留为标量或诊断，不默认投进日内 alpha。
- `price_limit`：是否已经触及涨停/跌停状态，是标签或交易约束，不是默认方向因子。

---

## 五、只控制仓位，不判方向

- `vpin`：按 BVC 和等成交量桶计算订单流毒性。高时说明市场更危险。
- `kyle_lambda`：原始 Kyle lambda 是单位成交量造成的价格冲击；仓位闸门会另行标准化。
- `exposure_gate`：把 VPIN 和 Kyle lambda 合成 0.2-1.0 的仓位系数。

---

## 六、合成与分组

`composite.py` 里三张表最重要:

1. `FACTOR_REGISTRY`: 每个因子的论文、公式、字段、角色和严格支持状态。
2. `DEFAULT_WEIGHTS`: 默认方向因子权重，总和为 1。
3. `FACTOR_GROUPS`: 可选分组：`alpha`, `flow`, `book`, `behavior`, `auction`, `limit`, `interaction`, `gate`, `diagnostic`, `label`。

示例：

```bash
python3 scripts/run_alpha_research.py --factors flow,auction
```

---

## 七、必须记住的诚实警告

1. 合成数据只能验证代码流程，不能证明赚钱。
2. 股票有 T+1，日内股票回转不现实；日内交易更适合期货。
3. 高频因子必须用真实 L2 数据做样本外检验。
4. 涨跌停、手续费、滑点、市场冲击会显著改变结果。

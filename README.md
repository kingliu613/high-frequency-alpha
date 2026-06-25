# High-Frequency Alpha — 项目总纲

## 愿景

让 Claude 全自动完成从论文到可用因子的完整链路，最终演化出人类研究员无法主动设计的新型因子。

---

## 两阶段路线图

### 第一阶段：自动化因子工厂

**目标：** 输入论文 → 输出可在 A 股回测的因子代码，全程无需人工干预。

```
论文 PDF / SSRN
    ↓  Claude 阅读、理解、提取信号公式
因子实现（Python）
    ↓  A 股数据适配（CSMAR / Tushare）
回测引擎（T+1、涨跌停、手续费、滑点）
    ↓  IC、Sharpe、最大回撤评估
因子库（标注论文来源、分类、评分）
```

核心能力需要逐步建立：

| 能力 | 当前状态 | 目标 |
|---|---|---|
| 论文解析 | 手动 | Claude 自动提取信号定义 |
| 因子实现 | 手写 | Claude 生成 Python 代码 |
| 数据适配 | CRSP/Compustat（美股） | CSMAR/Tushare（A 股） |
| 回测引擎 | 基础可用 | 更贴近真实：L2 数据、冲击成本 |
| 因子评估 | IC / Sharpe | 多维度打分 + 自动入库 |

**素材库积累目标：** 至少 200 个经过 A 股验证的因子（涵盖高频微观结构、日频截面、基本面）。

参考素材库：[OpenSourceAP/CrossSection](https://github.com/OpenSourceAP/CrossSection) — 319 个截面因子，有 Python 实现。

---

### 第二阶段：因子进化系统

**灵感：** AlphaGo（强化学习自我博弈）+ AlphaFold（结构空间随机搜索）+ 遗传算法。

**核心机制：**

```
因子基因库（第一阶段积累的 200+ 因子）
    ↓
随机生成新因子（算子组合、参数突变、跨因子杂交）
    ↓
快速回测筛选（剔除 IC < 阈值、Sharpe < 阈值）
    ↓
存活因子进入下一代基因库
    ↓
循环迭代 → 因子越来越强
```

**进化算子：**
- **突变**：改变时间窗口参数、加权方式、标准化方法
- **杂交**：组合两个不相关因子（线性、非线性）
- **变异**：替换信号中的某一数据字段（成交量 → 委托量）
- **剪枝**：与现有因子相关性 > 0.8 的直接淘汰（去冗余）

**适应度函数（回测评分）：**
- 样本外 IC_IR > 0.3
- 年化 Sharpe > 1.0（扣费后）
- 最大回撤 < 20%
- 与库内现有因子平均相关性 < 0.5（保多样性）

**终极目标：** 系统自主发现人类研究员在论文中从未描述过的新型因子结构。

---

## 当前进展（第一阶段）

### 已实现信号模块

| 模块 | 信号 | 数据层级 |
|---|---|---|
| `src/signals/ofi.py` | OFI、聚合 OFI、成交失衡 | 高频 L2 |
| `src/signals/features.py` | 队列失衡、涨跌停状态 | 高频 L2 |
| `src/signals/auction.py` | 集合竞价失衡 | 开盘 L2 |
| `src/signals/advanced.py` | API、OEI、羊群、撤单、风险闸门 | 高频 L2 |
| `src/signals/composite.py` | 因子注册表、权重、合成 | — |
| `src/signals/daily.py` | 日频聚合、截面评估 | 日频 |

### 已实现基础设施

- 合成数据生成器（`src/data/synthetic.py`）
- 真实数据 loader 框架（`src/data/loader.py`）
- 回测引擎：T+1、滑点、市场冲击、涨跌停约束

### 待完成（第一阶段关键缺口）

- [ ] A 股真实数据接入（CSMAR 或 Tushare Pro）
- [ ] CrossSection 319 因子 → A 股适配（变量映射）
- [ ] Claude 自动从 PDF 提取信号公式并生成代码
- [ ] 因子评分体系 + 自动入库流程

---

## 参考文献库

| 来源 | 内容 | 路径 |
|---|---|---|
| Chen & Zimmermann (2021) | 319 个截面因子复现 | `Documents/Caitong papers/ssrn-3604626 300+ alphas.pdf` |
| CrossSection GitHub | 319 因子 Python 实现 | `Documents/CrossSection/` |
| `research/hft_factors_china.md` | 已研究的高频因子目录 | 本项目 |
| `research/ALPHA_NOTES.md` | 研究笔记 | 本项目 |

---

## 技术栈

- **语言：** Python 3.10+
- **数据：** Tushare Pro / CSMAR（A 股）
- **回测：** 自建引擎（`src/backtest/`）
- **AI 驱动：** Claude API（论文解析、因子代码生成、进化算子）
- **存储：** Parquet（因子数据）+ JSON（因子元数据库）

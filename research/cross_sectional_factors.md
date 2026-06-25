# A股截面因子宇宙

来源：OpenSourceAP/CrossSection（Chen & Zimmermann 2021，319个因子，212个 clear+likely predictor）
A股可移植估计：约 **170个**

---

## 移植优先级总览

| 优先级 | 类别 | 因子数 | 数据需求 | A股可行性 |
|---|---|---|---|---|
| **P0** | 价格/收益类 | 45 | 日/月行情 | 极高 — Tushare 直接获取 |
| **P0** | 交易/流动性类 | 13 | 日行情+成交量 | 高 |
| **P1** | 会计基本面类 | 99 | CSMAR 财报 | 高（需字段映射） |
| **P2** | 分析师类 | 18 | Wind/CSMAR 分析师库 | 中（需付费数据） |
| **跳过** | 期权类 | 9 | 个股期权 | 极低 |
| **跳过** | 13F机构持仓类 | 8 | 13F披露 | 低 |

---

## P0 — 价格/收益类（45个）

### 动量 Momentum（11个）
| CrossSection名 | 经济含义 | A股注意事项 |
|---|---|---|
| Mom12m | 12个月动量（跳过最近1个月） | A股动量较弱，需检验 |
| Mom6m | 6个月动量 | 同上 |
| IndMom | 行业动量 | 需A股行业分类（申万/中信） |
| IntMom | 中期动量 | — |
| FirmAgeMom | 按上市年龄分组的动量 | 需IPO日期 |
| Mom12mOffSeason | 非同期月份动量 | — |
| MomOffSeason | 季节性动量（剔除同期月份） | — |
| MomOffSeason06YrPlus | 同上，6年以上 | — |
| MomOffSeason16YrPlus | 同上，16年以上 | — |
| MomSeason | 同月份历史收益（季节性） | — |
| MomVol | 动量×波动率交互 | — |

### 风险/Beta（6个）
| CrossSection名 | 经济含义 | A股注意事项 |
|---|---|---|
| Beta | 市场Beta（60个月） | 用沪深300作为市场组合 |
| BetaTailRisk | 尾部风险Beta | — |
| CoskewACX | 共偏度 | — |
| ReturnSkew | 收益偏度 | — |
| ReturnSkew3F | 三因子残差偏度 | 需三因子数据 |
| BetaLiquidityPS | Pastor-Stambaugh流动性Beta | 需A股流动性因子 |

### 波动率（5个）
| CrossSection名 | 经济含义 | A股注意事项 |
|---|---|---|
| IdioVol3F | 三因子残差波动率（特质波动率） | A股特质波动率效应显著 |
| IdioVolAHT | AHT方法特质波动率 | — |
| MaxRet | 近1个月最大单日收益 | — |
| RealizedVol | 已实现波动率 | — |
| betaVIX | VIX Beta | 可用A股VIX替代（iVIX） |

### 传导/价格发现 Lead-Lag（5个）
| CrossSection名 | 经济含义 | A股注意事项 |
|---|---|---|
| PriceDelaySlope | 价格发现延迟（斜率法） | A股信息效率低，效应强 |
| PriceDelayRsq | 价格发现延迟（R²法） | — |
| PriceDelayTstat | 价格发现延迟（t统计量法） | — |
| IndRetBig | 大股票收益对小股票的预测 | — |
| retConglomerate | 多元化集团内传导 | — |

### 反转（3个）
| CrossSection名 | 经济含义 |
|---|---|
| STreversal | 短期反转（1个月） |
| LRreversal | 长期反转（3-5年） |
| MRreversal | 中期反转（1-3年） |

### 其他价格类（15个）
季节性动量变体、公告收益、规模等。

---

## P0 — 交易/流动性类（13个）

| CrossSection名 | 经济含义 | A股数据来源 |
|---|---|---|
| Illiquidity | Amihud非流动性 | 日行情（价格冲击/成交额） |
| std_turn | 换手率波动率 | 日行情换手率 |
| VolSD | 成交量标准差 | — |
| zerotrade6M | 6个月零成交天数比例 | — |
| zerotrade1M | 1个月零成交天数比例 | — |
| BidAskSpread | 买卖价差（估算） | 日OHLC估算 |
| DolVol | 月度成交额 | — |
| ShareVol | 月度成交量 | — |
| VolMkt | 成交量/市场总量 | — |
| VolumeTrend | 成交量趋势 | — |
| ShortInterest | 融券余额 | **A股限制，慎用** |

> A股特有调整：换手率在A股异常高（零售驱动），需行业中性化后使用。

---

## P1 — 会计基本面类（99个）

### 估值 Valuation（15个）
| CrossSection名 | 经济含义 | Compustat→CSMAR映射 |
|---|---|---|
| BM | 账面市值比 | ceq/mktcap → 净资产/总市值 |
| BMdec | 12月账面市值比 | 同上，取12月财报 |
| AM | 资产市值比 | at/mktcap → 总资产/总市值 |
| CF | 现金流价格比 | ib+dp/mktcap → 净利+折旧/总市值 |
| EP | 盈利价格比 | ib/mktcap → 净利润/总市值 |
| SP | 销售价格比 | sale/mktcap → 营业收入/总市值 |
| DivSeason | 股息季节性 | — |
| ... | | |

### 盈利质量 Profitability（7个）
| CrossSection名 | 经济含义 |
|---|---|
| GP | 毛利率（Novy-Marx） |
| OperProf | 营业利润率（Fama-French） |
| roaq | 季度资产收益率 |
| CBOperProf | 现金营业盈利 |
| InvGrowth | 库存增长 |

### 投资 Investment（17个）
| 子类 | 代表因子 |
|---|---|
| 资产增长 | AssetGrowth、dNoa |
| 权益增长 | ChEQ、DelEqu |
| 应计 | Accruals、PctAcc、AbnormalAccruals |
| 净营运资本变化 | ChNWC、ChNNCOA |

### 外部融资 External Financing（11个）
CompEquIss（股权增发）、CompositeDebtIssuance（综合债务发行）等

### 综合打分 Composite（4个）
MS（Mohanram G-score）、PS（Piotroski F-score）、FR、RDS

---

## P2 — 分析师类（18个）

需要 Wind 或 CSMAR 分析师数据库（约2-5万/年）：

| 子类 | 代表因子 |
|---|---|
| 盈利预测修正 | AnalystRevision、REV6、ChForecastAccrual |
| 预测离散度 | EarningsForecastDisparity、ForecastDispersion |
| 评级变化 | ChangeInRecommendation、ConsRecomm |
| 长期增长预测 | fgr5yrLag |

---

## 跳过的因子

| 类别 | 原因 |
|---|---|
| Options（9个） | A股个股期权市场极不发达 |
| 13F持仓（8个） | A股机构披露季度，颗粒度不够，无直接等价物 |
| Short Interest（精细版） | A股融券成本高、规模小、数据不完整 |
| 部分Other（治理等） | 治理指数数据稀缺或不连续 |

---

## 数据来源对照表

| 用途 | 推荐来源 | 成本 |
|---|---|---|
| 日/月行情 | Tushare Pro | 低（约200积分/天） |
| 财务报表（年报/季报） | CSMAR | 学术免费 或 商业授权 |
| 分析师预测 | Wind 或 CSMAR 分析师库 | 较高 |
| 行业分类 | Tushare（申万行业） | 低 |
| 指数成分 | Tushare | 低 |

---

## 实施路线

```
第1步（P0价格类）：
  - 接 Tushare 日行情
  - 实现 Mom12m, IdioVol3F, Illiquidity, STreversal, Beta
  - 单因子IC检验（2015-2024，沪深300成分）

第2步（P0流动性类）：
  - Amihud非流动性、零成交天数、换手率波动
  - 行业中性化处理

第3步（P1会计类）：
  - 接 CSMAR 财报数据
  - Compustat→CSMAR 字段映射（见 src/data/csmar_loader.py）
  - BM, GP, OperProf, AssetGrowth, Accruals

第4步：
  - 因子入库（SQLite 或 Parquet）
  - 标准化评分体系（IC, ICIR, Sharpe, 最大回撤）
  - 自动入库流程
```

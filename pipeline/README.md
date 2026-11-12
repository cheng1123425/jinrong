# 研报复现流水线

把「读研报 → 写代码 → 看 IC 图 → 说有效」切成 8 个阶段，每个阶段有固定产物和代码化门禁。
PM 跑一条命令就能知道卡在哪，而不是靠回忆 25 条判据。

```
S0 规格 → S1 数据 → S2 单季化 → S3 时点★ → S4 计算 → S5 标准化 → S6 评价 → S7 报告
                                 ↑ 前视红线      ↑ σ 无偏      ↑ 成本必扣
```

## 快速开始

```bash
# 一条命令：生成合成数据 → 跑全流程 → 输出门禁结果
python src/run.py --demo --out runs/demo

# 用真实数据：改 spec 里的两个路径即可
python src/run.py --spec config/factor_spec.example.json --out runs/my_factor

# 门禁自测（反例注入，证明门禁不是摆设）
python tests/test_gates.py

# 拆解回测：抽一个调仓日，打印 IC / 分组 / 多空 / 换手的每一步中间量
python src/inspect_backtest.py --run runs/demo

# 谁来做：打印阶段 → 角色 → 专家 → 代码入口 → 门禁 → 人工复核项
python src/run.py --owners
```

## 骨架内容

```
pipeline/
├── PIPELINE.md              8 阶段 + 26 条门禁 + 三条红线 + 责任分工
├── 复核_骨架自查.md          文档声称 vs 代码实际：6 个缺陷与修复验证
├── pm/派单与验收.md          RACI、任务单/验收单模板、五级结论、变更留痕
├── config/
│   ├── factor_spec.example.json   因子规格（唯一真源）
│   └── owners.json          阶段 → 角色 → 专家 → 代码入口 绑定表
├── src/
│   ├── common.py            截面统计 / IC / NW-t / 分层 / 成本
│   ├── demo_data.py         合成数据（植入已知信号）
│   ├── stages.py            S0–S7 实现
│   ├── gates.py             26 条门禁
│   ├── inspect_backtest.py  单日拆解：IC/分层/多空/换手的中间量
│   └── run.py               入口（含 --owners / --force / --json）
├── tests/test_gates.py      11 个反例注入用例
└── runs/
    ├── demo_data/           prices.pkl + reports.pkl
    └── demo/                00~08 产物
```

## 演示运行结果（合成数据）

```
门禁汇总：PASS 26 / WARN 0 / FAIL 0
```

```
股票池  200 → 173（剔金融 16、ST 11）
IC 均值            0.032        ← 植入信号 3%/σ
ICIR               0.393
NW-t               3.18
独立截面数          65（5.4 年）
多空年化            4.74%
多空夏普            0.78
多空最大回撤        -5.28%
换手率（单边）      45.78%
扣费后多空年化      3.03%
分层收益  L1 -0.27% → L5 +0.20%（单调=False，两端方向正确）
```

这组数字的作用是**验证管线连通**：植入 3%/σ 的信号，跑出来 IC 0.031、t 3.26、扣费后仍为正，说明从单季化到时点对齐到回归再到评价这条链是通的。
**不能引用为真实市场结论**——数据是合成的，`SIGNAL` 是写死的常量。

## 谁来做

| 类型 | 承担者 | 内容 |
|---|---|---|
| 算 | 代码 | 全部计算 |
| 卡 | 代码 | 26 条门禁，禁止 agent 改判 |
| 定口径 | 专家 | 公式、去累计、回归设定、中性化 |
| 放行 | PM | 判 WARN、出验收结论 |

`python src/run.py --owners` 打印完整绑定表；`08_gates.json` 里每条门禁都带 `owner` / `expert` / `human_review`。

## 骨架解决的四个具体问题

| 问题 | 骨架怎么解 |
|---|---|
| 每次复现都从零搭代码 | 只改 spec，八阶段与门禁全部复用 |
| 前视靠人眼看 | G3.1 把 `align.mode != ann_date` 直接判 FAIL |
| σ 用错自由度，因子被放大 1.3–2.0 倍且无人发现 | G4.2 要求 `ddof ≥ 3`；G4.3 报放大系数 |
| 结论只有图、没有数，无法复核 | G6.1 要求 11 项指标无空白；G7.3 禁止 TODO/nan |

## 门禁也会被测试

`tests/test_gates.py` 故意把流水线改错，断言门禁必须拦住：

```
T1 正常路径 0 FAIL
T2 报告期末对齐 → G3.1 FAIL
T3 ddof=0 + min_n=4 → G4.1、G4.2 FAIL
T4 不中性化且无豁免 → G5.3 FAIL
T5 成本 0 → G0.2、G7.1 FAIL
T6 缺夏普 → G6.1 FAIL
T7 无单调性判定 → G6.4 FAIL
T8 成本占毛利过高 → G7.2 WARN
```

没有反例测试的门禁，可能只是一条永远 PASS 的空规则——那比没有门禁更危险。

## 已知边界

- **只有一个公式实现**（`ocfs_residual`）。接新因子需在 `stages.stage4_factor` 加分支，其余阶段自动复用。
- **数据格式是自定义的**：`prices.pkl` / `reports.pkl`。接真实数据要先做一层转换。
- **不含涨跌停/停牌可交易性过滤**。G7 只扣成本，没扣"买不进卖不出"。真实回测必须补。
- **不做样本外切分**。当前是一个区间跑完；要加 OOS 需要在 spec 里加 `evaluate.oos_split` 并在 S6 加分支。
- **门禁只管"有没有"，不管"对不对"**。公式是否等于研报定义、成本假设是否贴近现实，仍要人判。
- **26 条门禁里只有 9 条有反例测试**，其余 17 条只跑过正例——这是当前最大短板，见 `复核_骨架自查.md` 第四节。

# -*- coding: utf-8 -*-
"""门禁（gate）：把 PM 的验收标准写成代码，每阶段跑完立刻判定 PASS / WARN / FAIL。

设计原则：
1. 门禁只读产物，不改任何数据；
2. FAIL 必须给出「哪一条、当前值、阈值、怎么修」；
3. 宁严勿松：拿不准的给 WARN，但前视类一律 FAIL。
"""
import os

REQUIRED_METRICS = [
    "ic_mean", "icir", "nw_t", "ic_decay", "layer_returns", "ls_annual",
    "ls_sharpe", "ls_mdd", "turnover", "net_ls_annual", "n_periods",
]


def _r(gid, stage, name, status, detail, fix=""):
    return {"id": gid, "stage": stage, "name": name, "status": status,
            "detail": detail, "fix": fix}


def gate_s0(spec, art):
    out = []
    chk = art.get("spec_checks", {})
    miss = chk.get("missing_fields") or []
    out.append(_r("G0.1", "S0", "规格必填项完整",
                  "PASS" if not miss else "FAIL",
                  "全部齐备" if not miss else f"缺 {len(miss)} 项：{'; '.join(miss)}",
                  "补齐 config/factor_spec 后重跑"))
    out.append(_r("G0.2", "S0", "调仓/持有/成本已声明",
                  "PASS" if chk.get("cost_bps") not in (None, 0) else "FAIL",
                  f"cost_bps={chk.get('cost_bps')}",
                  "成本必须为正，否则扣费结论无意义"))
    return out


def gate_s1(spec, art):
    c = art.get("coverage")
    out = []
    if not c:
        return [_r("G1.0", "S1", "覆盖率报告存在", "FAIL", "缺少 01_coverage.json", "重跑 S1")]
    out.append(_r("G1.1", "S1", "股票数与公告日数足够",
                  "PASS" if c["n_codes_price"] >= 50 and c["n_ann_dates"] >= 8 else "FAIL",
                  f"{c['n_codes_price']} 只 × {c['n_ann_dates']} 个公告日",
                  "样本过少，任何统计都不可靠"))
    share = c.get("null_cfo_share", 0)
    out.append(_r("G1.2", "S1", "核心字段缺失率已披露且可接受",
                  "PASS" if share <= 0.10 else "WARN",
                  f"cfo 缺失率 {share*100:.2f}%（阈值 10%）",
                  "缺失率高时需在报告中说明丢弃机制与影响"))
    u = art.get("universe") or {}
    applied = (u.get("applied") or {})
    want = (spec.get("universe") or {})
    mismatch = []
    for key in ("exclude_financial", "exclude_st"):
        if key in want and bool(want.get(key)) != bool(applied.get(key)):
            mismatch.append(f"{key}: spec={want.get(key)} / 实际={applied.get(key)}")
    out.append(_r("G1.3", "S1", "股票池过滤按 spec 执行且已披露",
                  "PASS" if not mismatch and u.get("dropped") is not None else "FAIL",
                  "剔除 " + "/".join(f"{k} {v}" for k, v in (u.get("dropped") or {}).items())
                  if not mismatch else "; ".join(mismatch),
                  "spec 写了但代码没执行，等于股票池未定义"))
    return out


def gate_s2(spec, art):
    q = art.get("quarterly_report")
    out = []
    if not q:
        return [_r("G2.0", "S2", "单季化产物存在", "FAIL", "缺少 02_quarterly.pkl", "重跑 S2")]
    out.append(_r("G2.1", "S2", "去累计口径已写明",
                  "PASS" if q.get("de_cumulate") else "FAIL",
                  q.get("de_cumulate", "未写明"),
                  "必须说明 Q1 是否取累计、跨年是否重置"))
    v = q.get("cfo_valid_share", 0)
    out.append(_r("G2.2", "S2", "单季 CFO 有效率",
                  "PASS" if v >= 0.80 else "WARN",
                  f"{v*100:.1f}%（阈值 80%）",
                  "有效率低说明原始累计值缺口大，需检查数据源"))
    return out


def gate_s3(spec, art):
    a = art.get("asof_report")
    out = []
    if not a:
        return [_r("G3.0", "S3", "时点报告存在", "FAIL", "缺少 03_asof_report.json", "重跑 S3")]
    mode = a.get("mode")
    out.append(_r("G3.1", "S3", "必须用公告日对齐（前视红线）",
                  "PASS" if mode == "ann_date" else "FAIL",
                  f"align.mode={mode}",
                  "改成 ann_date；用报告期末对齐会让因子提前 2-4 个月可见"))
    v = a.get("violations_date_before_period", 0)
    out.append(_r("G3.2", "S3", "因子日期不得早于报告期",
                  "PASS" if v == 0 else "FAIL",
                  f"违规 {v} 条",
                  "逐条核对 ann_date；这是最容易被忽略的隐性前视"))
    return out


def gate_s4(spec, art):
    d = art.get("factor_diag")
    out = []
    if not d:
        return [_r("G4.0", "S4", "因子诊断存在", "FAIL", "缺少 04_factor_diag.json", "重跑 S4")]
    out.append(_r("G4.1", "S4", "最小样本 ≥ 6（自由度 ≥ 3）",
                  "PASS" if (d.get("min_n") or 0) >= 6 else "FAIL",
                  f"min_n={d.get('min_n')}，残差自由度 = {max((d.get('min_n') or 0)-3, 0)}",
                  "min_n=4 时 σ 由 1 个自由度估计，因子被放大 2 倍"))
    out.append(_r("G4.2", "S4", "σ 无偏（ddof ≥ 3，即扣掉三个参数）",
                  "PASS" if (d.get("ddof") or 0) >= 3 else "FAIL",
                  f"ddof={d.get('ddof')}，无偏值应为 3",
                  "np.std 除以 (n-d)；三参数回归需 d=3，否则 σ 被系统性低估"))
    inf = d.get("sigma_inflation")
    out.append(_r("G4.3", "S4", "σ 放大系数接近 1",
                  "PASS" if (inf is not None and inf <= 1.15) else "WARN",
                  f"√(n̄/(n̄−3)) = {inf:.3f}" if inf else "未计算",
                  ">1.15 说明分母仍被低估，需提高 min_n 或改 pooled σ"))
    b, bs = d.get("beta_mean"), d.get("beta_sd")
    out.append(_r("G4.4", "S4", "β 稳定（不爆炸）",
                  "PASS" if (b is not None and abs(b) <= 1.5 and (bs or 0) <= 0.8) else "WARN",
                  f"β 均值 {b:.3f}，σ {bs:.3f}" if b is not None else "未计算",
                  "β 离 1 太远或跨股票方差过大，说明估计被少数期主导"))
    out.append(_r("G4.5", "S4", "样本量分布已披露",
                  "PASS" if d.get("n_clean_median") is not None else "FAIL",
                  f"中位 {d.get('n_clean_median')}，P10 {d.get('n_clean_p10')}",
                  "不报分布就无法判断排序是否混入数据完整度"))
    return out


def gate_s5(spec, art):
    p = art.get("preprocess")
    out = []
    if not p:
        return [_r("G5.0", "S5", "预处理报告存在", "FAIL", "缺少 05_preprocess.json", "重跑 S5")]
    out.append(_r("G5.1", "S5", "极值处理已开启",
                  "PASS" if p.get("winsorize") else "FAIL",
                  f"winsorize={p.get('winsorize')}",
                  "不做极值处理，单只异常值会主导分层"))
    out.append(_r("G5.2", "S5", "截面标准化已开启",
                  "PASS" if p.get("rank") else "FAIL",
                  f"rank={p.get('rank')}",
                  "不同股票的残差尺度不同，不做截面标准化不可比"))
    waiver = (spec.get("notes") or {}).get("neutralize_waiver")
    out.append(_r("G5.3", "S5", "行业/规模中性化（或有书面豁免）",
                  "PASS" if p.get("neutralize") or waiver else "FAIL",
                  f"neutralize={p.get('neutralize')}" + (f"，豁免理由：{waiver}" if waiver else ""),
                  "不中性化则无法排除排序主要来自行业与市值暴露"))
    keep = (p.get("n_after") or 0) / max(p.get("n_before") or 1, 1)
    out.append(_r("G5.4", "S5", "预处理未过度丢样本",
                  "PASS" if keep >= 0.90 else "WARN",
                  f"保留 {keep*100:.1f}%（阈值 90%）",
                  "丢样本过多会让后期截面变薄"))
    return out


def gate_s6(spec, art):
    m = art.get("metrics")
    out = []
    if not m:
        return [_r("G6.0", "S6", "指标文件存在", "FAIL", "缺少 06_metrics.json", "重跑 S6")]
    missing = [k for k in REQUIRED_METRICS if m.get(k) is None]
    out.append(_r("G6.1", "S6", "11 项核心指标无空白",
                  "PASS" if not missing else "FAIL",
                  "齐备" if not missing else f"缺：{', '.join(missing)}",
                  "逐项补齐；缺一项即无法复核"))
    n = m.get("n_periods") or 0
    out.append(_r("G6.2", "S6", "独立截面数 ≥ 20",
                  "PASS" if n >= 20 else "WARN",
                  f"{n} 个截面（≈{m.get('n_independent_years')} 年）",
                  "截面太少时 ICIR 与 t 值都不可靠"))
    t = m.get("nw_t")
    out.append(_r("G6.3", "S6", "已做重叠调整（NW-t）",
                  "PASS" if t is not None else "FAIL",
                  f"NW-t={t:.2f}" if t is not None else "未计算",
                  "调仓期内 IC 序列重叠，朴素 t 会虚高数倍"))
    out.append(_r("G6.4", "S6", "分层单调性已程序判定",
                  "PASS" if m.get("layer_monotonic") is not None else "FAIL",
                  f"layer_monotonic={m.get('layer_monotonic')}",
                  "不允许用文字描述'近乎完美'替代判定"))
    return out


def gate_s7(spec, art):
    m = art.get("metrics") or {}
    out = []
    out.append(_r("G7.1", "S7", "成本已扣",
                  "PASS" if (m.get("cost_bps") or 0) > 0 and m.get("net_ls_annual") is not None else "FAIL",
                  f"cost_bps={m.get('cost_bps')}，扣费后年化={m.get('net_ls_annual')}",
                  "未扣费的多空收益不能作为有效性证据"))
    cost = m.get("cost_per_rebalance") or 0
    gross = m.get("ls_annual") or 0
    spacing = float(m.get("rebalance_days") or 21)
    out.append(_r("G7.2", "S7", "成本未吃掉一半以上收益",
                  "PASS" if gross == 0 or cost * (252 / max(spacing, 1)) <= abs(gross) * 0.5 else "WARN",
                  f"每期成本 {cost*100:.3f}%，年化毛利 {gross*100:.2f}%",
                  "成本占比过高时策略不可实施，需降低换手"))
    p = art.get("report_path")
    ok = bool(p and os.path.exists(p))
    txt = ""
    if ok:
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
    out.append(_r("G7.3", "S7", "报告已生成且无占位符",
                  "PASS" if ok and "TODO" not in txt and "nan%" not in txt else "FAIL",
                  p or "无报告",
                  "报告必须由指标文件程序化生成"))
    need = ["## 1.", "## 2.", "## 3.", "## 4.", "## 5."]
    out.append(_r("G7.4", "S7", "报告五节齐全（数据/时点/估计/表现/结论）",
                  "PASS" if ok and all(s in txt for s in need) else "FAIL",
                  "齐全" if ok and all(s in txt for s in need) else "章节缺失",
                  "缺任何一节都不可验收"))
    return out


GATES = {0: gate_s0, 1: gate_s1, 2: gate_s2, 3: gate_s3,
         4: gate_s4, 5: gate_s5, 6: gate_s6, 7: gate_s7}


def run_gate(stage, spec, art):
    fn = GATES.get(stage)
    return fn(spec, art) if fn else []


def summarize(results):
    return {
        "PASS": sum(1 for r in results if r["status"] == "PASS"),
        "WARN": sum(1 for r in results if r["status"] == "WARN"),
        "FAIL": sum(1 for r in results if r["status"] == "FAIL"),
    }

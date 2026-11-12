# -*- coding: utf-8 -*-
"""门禁反例注入测试：证明门禁真的能拦住错误，而不是摆设。

每个用例故意把流水线改成错的，断言对应门禁必须 FAIL。
如果某个用例没拦住，说明该门禁是空的。

用法：python tests/test_gates.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import stages          # noqa: E402
import gates           # noqa: E402
from run import run_pipeline  # noqa: E402

TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_test")
N = 0
FAILED = 0


def check(name, cond, detail=""):
    global N, FAILED
    N += 1
    if not cond:
        FAILED += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def run_with(mutate, tag):
    spec = copy.deepcopy(stages.default_spec())
    mutate(spec)
    r = run_pipeline(spec, os.path.join(TMP, tag), stop_on_fail=False, verbose=False)
    return r["gates"]


def failed_ids(g):
    return {x["id"] for x in g["results"] if x["status"] == "FAIL"}


def main():
    print("=== 门禁反例注入测试 ===\n")

    # T1 正常路径：不应有 FAIL
    g = run_with(lambda s: None, "t1_happy")
    check("T1 正常路径 0 FAIL", g["summary"]["FAIL"] == 0,
          f"PASS {g['summary']['PASS']} / WARN {g['summary']['WARN']} / FAIL {g['summary']['FAIL']}")

    # T2 时点改报告期末 → 必须被 G3.1 拦住
    g = run_with(lambda s: s["align"].__setitem__("mode", "period_end"), "t2_asof")
    f = failed_ids(g)
    check("T2 报告期末对齐 → G3.1 FAIL", "G3.1" in f, f"失败项 {sorted(f)}")

    # T3 σ 用 ddof=0 且最小样本 4 → 必须被 G4.1 / G4.2 拦住
    def m3(s):
        s["formula"]["ddof"] = 0
        s["formula"]["min_n"] = 4
    g = run_with(m3, "t3_sigma")
    f = failed_ids(g)
    check("T3 ddof=0 + min_n=4 → G4.1 与 G4.2 FAIL",
          {"G4.1", "G4.2"}.issubset(f), f"失败项 {sorted(f)}")

    # T4 关掉中性化且无豁免 → 必须被 G5.3 拦住
    g = run_with(lambda s: s["preprocess"].__setitem__("neutralize", False), "t4_neutral")
    f = failed_ids(g)
    check("T4 不中性化且无豁免 → G5.3 FAIL", "G5.3" in f, f"失败项 {sorted(f)}")

    # T4b 关掉中性化但有书面豁免 → 应放行
    def m4b(s):
        s["preprocess"]["neutralize"] = False
        s["notes"] = {"neutralize_waiver": "本因子已在构造中行业内排序"}
    g = run_with(m4b, "t4b_waiver")
    check("T4b 有豁免 → G5.3 不 FAIL", "G5.3" not in failed_ids(g))

    # T5 成本设 0 → 必须被 G0.2 拦住
    g = run_with(lambda s: s["evaluate"].__setitem__("cost_bps", 0), "t5_cost0")
    f = failed_ids(g)
    check("T5 成本 0 → G0.2 FAIL", "G0.2" in f, f"失败项 {sorted(f)}")

    # T6 指标缺项 → 直接喂残缺 metrics 给 G6
    art = {"metrics": {"ic_mean": 0.03, "icir": 0.4, "nw_t": 3.1, "ic_decay": {},
                       "layer_returns": [0.1] * 5, "ls_annual": 0.05,
                       "ls_sharpe": 1.0, "ls_mdd": -0.06,
                       "turnover": 0.4, "net_ls_annual": 0.03, "n_periods": 60}}
    del art["metrics"]["ls_sharpe"]
    res = gates.gate_s6(stages.default_spec(), art)
    ids = {x["id"] for x in res if x["status"] == "FAIL"}
    check("T6 缺夏普 → G6.1 FAIL", "G6.1" in ids, f"失败项 {sorted(ids)}")

    # T7 分层单调用文字代替判定 → G6.4 必须拦
    art2 = {"metrics": dict(art["metrics"], ls_sharpe=1.0)}
    art2["metrics"].pop("layer_monotonic", None)
    res = gates.gate_s6(stages.default_spec(), art2)
    ids = {x["id"] for x in res if x["status"] == "FAIL"}
    check("T7 无 layer_monotonic → G6.4 FAIL", "G6.4" in ids, f"失败项 {sorted(ids)}")

    # T8 成本吃掉大部分收益 → 应给 WARN 而非 PASS
    art3 = {"metrics": dict(art2["metrics"]),
            "report_path": ""}
    art3["metrics"]["layer_monotonic"] = True
    art3["metrics"]["ls_annual"] = 0.02
    art3["metrics"]["cost_bps"] = 15
    art3["metrics"]["cost_per_rebalance"] = 0.004
    art3["metrics"]["rebalance_days"] = 21
    res = gates.gate_s7(stages.default_spec(), art3)
    st = {x["id"]: x["status"] for x in res}
    check("T8 成本占毛利过高 → G7.2 WARN", st.get("G7.2") == "WARN", f"{st}")

    # T9 判据从 spec 读：把 ic_min 提到 0.5，报告结论必须变成"不达标"
    def m9(s):
        s["criteria"] = {"ic_min": 0.5, "t_min": 2.0}
    g = run_with(m9, "t9_criteria")
    rp = os.path.join(TMP, "t9_criteria", "07_report.md")
    txt = open(rp, encoding="utf-8").read() if os.path.exists(rp) else ""
    check("T9 spec.criteria 生效（ic_min=0.5 → 报告判不达标）",
          "低于 0.500" in txt, "报告未反映 spec 判据" if "低于 0.500" not in txt else "")

    # T10 股票池门禁：spec 要求剔 ST 但实际没执行 → G1.3 必须 FAIL
    spec10 = stages.default_spec()
    spec10["universe"]["exclude_st"] = True
    art10 = {"coverage": {"n_codes_price": 200, "n_ann_dates": 28},
             "universe": {"applied": {"exclude_financial": True, "exclude_st": False},
                          "dropped": {"financial": 16, "st": 0, "short_history": 0}}}
    res10 = gates.gate_s1(spec10, art10)
    ids10 = {x["id"] for x in res10 if x["status"] == "FAIL"}
    check("T10 spec 要求剔 ST 但未执行 → G1.3 FAIL", "G1.3" in ids10, f"失败项 {sorted(ids10)}")

    print(f"\n结果：{N - FAILED}/{N} 通过" + ("" if FAILED == 0 else f"，{FAILED} 项失败"))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

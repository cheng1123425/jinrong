# -*- coding: utf-8 -*-
"""流水线入口。

用法：
  python src/run.py --demo                       # 生成合成数据 → 全流程 → 门禁
  python src/run.py --spec config/factor_spec.example.json --out runs/myfactor
  python src/run.py --demo --force               # 门禁 FAIL 也继续（仅用于看错误长什么样）
  python src/run.py --demo --json                # 只输出门禁汇总 JSON
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, load_json  # noqa: E402
import stages  # noqa: E402
import gates  # noqa: E402
import demo_data  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OWNERS_PATH = os.path.join(BASE, "config", "owners.json")


def load_owners():
    if os.path.exists(OWNERS_PATH):
        with open(OWNERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stages": {}}


STAGE_NAMES = {
    0: "S0 规格校验", 1: "S1 数据接入", 2: "S2 单季化", 3: "S3 时点对齐",
    4: "S4 因子计算", 5: "S5 截面标准化", 6: "S6 评价", 7: "S7 报告",
}
FUNCS = {
    0: stages.stage0_spec, 1: stages.stage1_ingest, 2: stages.stage2_quarterize,
    3: stages.stage3_asof, 4: stages.stage4_factor, 5: stages.stage5_standardize,
    6: stages.stage6_evaluate, 7: stages.stage7_report,
}


def _print_results(stage, results):
    if not results:
        return
    for r in results:
        mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[r["status"]]
        print(f"  [{mark}] {r['id']:<5} {r['name']}")
        if r["status"] != "PASS":
            print(f"         → {r['detail']}")
            if r.get("fix"):
                print(f"         → 修法：{r['fix']}")


def run_pipeline(spec=None, out_dir="runs/demo", stop_on_fail=True, verbose=True):
    spec = spec or stages.default_spec()
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(BASE, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    owners = load_owners()
    art = {}
    all_results = []
    failed_at = None

    for st in range(8):
        info = (owners.get("stages") or {}).get(f"S{st}", {})
        if verbose:
            who = info.get("owner_role", "未分配")
            exp = info.get("expert", "—")
            print(f"\n=== {STAGE_NAMES[st]} ｜ 负责：{who}（{exp}）===")
        FUNCS[st](spec, art, out_dir)
        res = gates.run_gate(st, spec, art)
        for r in res:
            r["stage_name"] = STAGE_NAMES[st]
            r["owner"] = info.get("owner_role", "未分配")
            r["expert"] = info.get("expert", "—")
            r["human_review"] = info.get("human_review", "")
        all_results.extend(res)
        if verbose:
            _print_results(st, res)
        fails = [r for r in res if r["status"] == "FAIL"]
        if fails and stop_on_fail:
            failed_at = st
            if verbose:
                print(f"\n⛔ 门禁在第 {st} 阶段拦截，共 {len(fails)} 条 FAIL。后续阶段不执行。")
            break

    summary = gates.summarize(all_results)
    payload = {"failed_at_stage": failed_at, "summary": summary,
               "owners": owners, "results": all_results}
    save_json(payload, os.path.join(out_dir, "08_gates.json"))

    if verbose:
        print("\n" + "=" * 62)
        print(f"门禁汇总：PASS {summary['PASS']} / WARN {summary['WARN']} / FAIL {summary['FAIL']}")
        if failed_at is not None:
            print(f"结论：不通过（第 {failed_at} 阶段 FAIL）")
        elif summary["FAIL"] == 0:
            print("结论：通过（可进入下一环节）；WARN 需在执行记录里写明处理方式")
        print(f"产物目录：{out_dir}")
    return {"artifacts": art, "gates": payload, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--demo", action="store_true", help="生成合成数据并跑全流程")
    ap.add_argument("--force", action="store_true", help="门禁 FAIL 也继续")
    ap.add_argument("--json", action="store_true", help="只打印门禁汇总 JSON")
    ap.add_argument("--owners", action="store_true", help="打印「谁做什么」责任矩阵后退出")
    a = ap.parse_args()

    if a.owners:
        own = load_owners()
        print("阶段 ｜ 负责角色 ｜ 绑定专家 ｜ 代码入口 ｜ 门禁 ｜ 人工复核项")
        print("-" * 130)
        for k, v in (own.get("stages") or {}).items():
            print(f"{k} {v['name']:<10} ｜ {v['owner_role']:<12} ｜ {v['expert']:<48} ｜ "
                  f"{v['code']:<26} ｜ {','.join(v['gates']):<22} ｜ {v['human_review']}")
        print("\n不外包给 agent 的：" + " / ".join(own.get("not_delegated") or []))
        sys.exit(0)

    if a.demo:
        demo_data.build(os.path.join(BASE, "runs", "demo_data"))

    if a.spec:
        spec = load_json(a.spec if os.path.isabs(a.spec) else os.path.join(BASE, a.spec))
    else:
        spec = stages.default_spec()

    r = run_pipeline(spec, a.out, stop_on_fail=not a.force, verbose=not a.json)
    if a.json:
        print(json.dumps(r["gates"], ensure_ascii=False, indent=2))
    sys.exit(0 if r["gates"]["summary"]["FAIL"] == 0 else 1)


if __name__ == "__main__":
    main()

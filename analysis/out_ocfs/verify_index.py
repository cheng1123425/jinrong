# -*- coding: utf-8 -*-
"""
T-OCFS-01 代码索引核对（沙盒复算）
目的：用合成数据把 __cal_resi__ 的切片展开成下标，验证 train_y/train_x1/train_x2 是否同期对齐。
说明：本脚本的数值全部来自我自己构造的合成 DGP，与文章结果无关，不得当作文章数值引用。
"""
import numpy as np

# ---------------- 原文代码的等价实现（逐字照抄切片逻辑） ----------------
def cal_resi_ref(y, x, debug=False):
    if len(y) < 4 or len(x) < 4 or np.isnan(y[0]) or np.isnan(y[1]) or np.isnan(x[0]):
        return np.nan
    max_samples = min(8, len(y) - 1)
    train_y  = y[1:1 + max_samples]
    train_x1 = y[2:2 + max_samples]
    train_x2 = x[1:1 + max_samples]
    train_data = np.column_stack([train_y, train_x1, train_x2])
    train_data_clean = train_data[~np.isnan(train_data).any(axis=1)]
    if train_data_clean.shape[0] < 4:
        return np.nan
    yc, x1c, x2c = train_data_clean[:, 0], train_data_clean[:, 1], train_data_clean[:, 2]
    X = np.column_stack([np.ones(len(yc)), x1c, x2c])
    coefs, _, _, _ = np.linalg.lstsq(X, yc, rcond=None)
    alpha, beta, gamma = coefs
    resid_in = yc - X @ coefs
    sigma = np.std(resid_in, ddof=0)
    if sigma == 0:
        return np.nan
    pred_x1 = y[1]
    pred_x2 = x[0]
    y_pred = alpha + beta * pred_x1 + gamma * pred_x2
    residual = y[0] - y_pred
    o_factor = residual / sigma
    if debug:
        return dict(n=max_samples, n_clean=len(yc), alpha=alpha, beta=beta, gamma=gamma,
                    sigma=sigma, pred_x1=pred_x1, pred_x2=pred_x2,
                    y_pred=y_pred, residual=residual, factor=o_factor)
    return o_factor


# ---------------- 测试 A：合成 DGP 下的系数回收 ----------------
# 设绝对期号 1..10，真实模型 CFO_s = a + b*CFO_{s-1} + c*dR_s + e_s
rng = np.random.default_rng(20261112)
a, b, c = 5.0, 0.6, -0.8
P = 11
cfo = np.zeros(P); dr = np.zeros(P)
cfo[1] = 10.0
for s in range(2, P):
    dr[s] = rng.normal(0, 2.0)
    cfo[s] = a + b * cfo[s-1] + c * dr[s] + rng.normal(0, 0.4)

T = P - 1                      # 最新期
y = np.array([cfo[T - k] for k in range(10)])   # k=0..9 -> CFO_T..CFO_{T-9}
x = np.array([dr[T - k] for k in range(9)])     # k=0..8 -> dR_T..dR_{T-8}

d = cal_resi_ref(y, x, debug=True)
print("=== 测试 A：对齐检验（n=8，完整样本） ===")
print(f"n={d['n']}, n_clean={d['n_clean']}")
print(f"回收系数: alpha={d['alpha']:.4f}, beta={d['beta']:.4f}, gamma={d['gamma']:.4f}")
print(f"真值    : alpha={a:.4f}, beta={b:.4f}, gamma={c:.4f}")
print(f"pred_x1=y[1]=CFO_(T-1)? {np.isclose(d['pred_x1'], cfo[T-1])}")
print(f"pred_x2=x[0]=dR_T?      {np.isclose(d['pred_x2'], dr[T])}")
# 用真系数直接外推 T 期，与代码的 y_pred 对比
manual = a + b * cfo[T-1] + c * dr[T]
print(f"代码 y_pred={d['y_pred']:.4f} / 真系数外推={manual:.4f} / 差={abs(d['y_pred']-manual):.4f}")
print()

# ---------------- 测试 A2：若 train_x1 少移一期（错误写法）会怎样 ----------------
def cal_resi_wrong(y, x):
    n = min(8, len(y) - 1)
    ty, tx1, tx2 = y[1:1+n], y[1:1+n], x[1:1+n]   # 故意让 x1 与 y 同期
    D = np.column_stack([ty, tx1, tx2])
    D = D[~np.isnan(D).any(axis=1)]
    X = np.column_stack([np.ones(len(D)), D[:, 1], D[:, 2]])
    coefs, *_ = np.linalg.lstsq(X, D[:, 0], rcond=None)
    return coefs

w = cal_resi_wrong(y, x)
print("=== 测试 A2：若漏掉一期移位（反例） ===")
print(f"错误写法系数: alpha={w[0]:.4f}, beta={w[1]:.4f}, gamma={w[2]:.4f}  <- beta 应贴 1.0，退化为恒等式")
print()

# ---------------- 测试 B：n 的三种情形展开 ----------------
print("=== 测试 B：n 与下标展开（长度按真实切片计） ===")
for tag, L in [("L=10(实际)", 10), ("L=9", 9), ("L=8", 8)]:
    n = min(8, L - 1)
    yy = np.arange(L, dtype=float); xx = np.arange(9, dtype=float)
    sy, sx1, sx2 = yy[1:1+n], yy[2:2+n], xx[1:1+n]
    ok = (len(sy) == len(sx1) == len(sx2))
    print(f"{tag}: n={n} | train_y=y[1:{1+n}] len={len(sy)} | train_x1=y[2:{2+n}] len={len(sx1)} "
          f"| train_x2=x[1:{1+n}] len={len(sx2)} -> 可堆叠={ok}")
print()
for n in [4, 5, 6, 7, 8]:
    print(f"n={n}: 行j (j=0..{n-1}) 依次取 (y[1+j], y[2+j], x[1+j])"
          f" -> 首行(y[1],y[2],x[1]) 末行(y[{n}],y[{n+1}],x[{n}])")
print()

# ---------------- 测试 A3：零噪声下系数能否精确回收（对齐的最强证据） ----------------
a0, b0, c0 = 5.0, 0.6, -0.8
cfo0 = np.zeros(P); dr0 = np.zeros(P); cfo0[1] = 10.0
for s in range(2, P):
    dr0[s] = 0.3 * s - 1.0
    cfo0[s] = a0 + b0 * cfo0[s-1] + c0 * dr0[s]
y0 = np.array([cfo0[T - k] for k in range(10)])
x0 = np.array([dr0[T - k] for k in range(9)])
d0 = cal_resi_ref(y0, x0, debug=True)
print("=== 测试 A3：零噪声精确回收（对齐成立则系数应精确等于真值，残差=0） ===")
print(f"回收: alpha={d0['alpha']:.6f}, beta={d0['beta']:.6f}, gamma={d0['gamma']:.6f}")
print(f"真值: alpha={a0:.6f}, beta={b0:.6f}, gamma={c0:.6f}")
print(f"样本内残差最大值={np.max(np.abs(np.array([cfo0[T-k] for k in range(1,9)])
      - (d0['alpha'] + d0['beta']*np.array([cfo0[T-k-1] for k in range(1,9)])
         + d0['gamma']*np.array([dr0[T-k] for k in range(1,9)])))):.3e}")
print()

# ---------------- 测试 C：len(y)<10 时 column_stack 报错 ----------------
print("=== 测试 C：面板不足 10 季时的行为 ===")
for L in [10, 9, 8]:
    yy = y[:L]; xx = x[:min(9, L - 1)]
    try:
        r = cal_resi_ref(yy, xx)
        print(f"len(y)={L} -> 返回 {r}")
    except Exception as e:
        print(f"len(y)={L} -> 抛异常 {type(e).__name__}: {e}")
print()

# ---------------- 测试 D：NaN 行删除是否破坏对齐 ----------------
print("=== 测试 D：NaN 删除后的行对齐 ===")
yy = y.copy(); xx = x.copy()
yy[3] = np.nan
r = cal_resi_ref(yy, xx, debug=True)
print(f"y[3] 置 NaN 后 n_clean={d['n_clean']} -> {r['n_clean']}，因子={r['factor']:.4f}（行级删除，未错行）")
print()

# ---------------- 测试 E：sigma 的自自由度偏差与放大倍数 ----------------
print("=== 测试 E：sigma 放大倍数（解析值 sqrt(n/(n-3))） ===")
for n in range(4, 9):
    print(f"n={n}: 残差自由度={n-3}, E[sigma_hat^2]=({n-3}/{n})sigma^2, "
          f"因子被放大 sqrt(n/(n-3))={np.sqrt(n/(n-3)):.3f}x")
print()

# 蒙特卡洛验证：8 个点、3 参数
rng2 = np.random.default_rng(7)
ratios = []
for _ in range(3000):
    n = 8
    Z = rng2.normal(size=(n, 2))
    Xd = np.column_stack([np.ones(n), Z])
    e = rng2.normal(0, 1.0, n)
    coef, *_ = np.linalg.lstsq(Xd, Xd @ np.array([1., 2., -1.]) + e, rcond=None)
    s = np.std(Xd @ np.array([1., 2., -1.]) + e - Xd @ coef, ddof=0)
    ratios.append(s)
print(f"MC(n=8, sigma_true=1.0): mean(sigma_hat)={np.mean(ratios):.4f}, "
      f"理论 sqrt(5/8)={np.sqrt(5/8):.4f}")
print()

# ---------------- 测试 F：n=4 时的因子爆炸 ----------------
print("=== 测试 F：n_clean=4 时的因子量级 ===")
for noise in [1.0, 0.05, 1e-3]:
    rng3 = np.random.default_rng(11)
    z = rng3.normal(size=(4, 2))
    Xd = np.column_stack([np.ones(4), z])
    tgt = Xd @ np.array([1., 2., -1.]) + rng3.normal(0, noise, 4)
    coef, *_ = np.linalg.lstsq(Xd, tgt, rcond=None)
    s = np.std(tgt - Xd @ coef, ddof=0)
    print(f"  残差噪声={noise:g} -> sigma_hat={s:.6f}, 若 |residual|~1 则 |factor|~{1/max(s,1e-12):.1f}")

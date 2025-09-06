import numpy as np
import matplotlib.pyplot as plt


def generate_q(J=12, mode="poisson", seed=0):
    rng = np.random.default_rng(seed)
    if mode == "poisson":
        lam = rng.poisson(3, size=J)
        q = 1.0 - np.exp(-lam / 5.0)
    elif mode == "normal":
        z = rng.normal(0, 2, size=J)
        q = 1.0 / (1.0 + np.exp(-z))
    elif mode == "gamma":
        z = rng.gamma(shape=2.0, scale=1.0, size=J)
        q = z / (z + 5.0)
    else:
        raise ValueError("mode must be poisson/normal/gamma")
    return np.clip(q, 1e-6, 1 - 1e-6)

def synthesize(A, q, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    # p_fail(i) = 1 - Π_j (1 - A_ij q_j)
    p_fail = 1.0 - np.prod(1.0 - A * q, axis=1)
    y = rng.binomial(1, np.clip(p_fail, 1e-9, 1 - 1e-9))
    return y, p_fail

def noisyor_em(A, y, a0=1.0, b0=8.0, max_iter=200, tol=1e-6, verbose=False):
    """
    A: (N,J) ∈ {0,1}, y: (N,) ∈ {0,1}
    回傳：q_est (J,), resp (N,J) 只在 y=1 的列非零, loglik_hist
    """
    N, J = A.shape
    rng = np.random.default_rng(0)

    # 初始化：用樣本失敗率或小常數
    q = np.full(J, max(0.01, y.mean()))
    q = np.clip(q, 1e-4, 1 - 1e-4)

    # 每個樣本經過的節點索引（加速）
    idx_per_i = [np.where(A[i] == 1)[0] for i in range(N)]
    loglik_hist = []

    for it in range(max_iter):
        
        resp = np.zeros((N, J))
        log1mq = np.log(1 - q + 1e-12)  # 常用於數值穩定
        for i in range(N):
            if y[i] == 0: 
                continue
            Js = idx_per_i[i]
            if Js.size == 0:
                continue
            # fail_prob = 1 - Π_{j∈Js}(1 - q_j)；用 log 
            log_survive = np.sum(log1mq[Js])
            fail_prob = 1 - np.exp(log_survive)
            if fail_prob <= 0: 
                continue
            # numer_j = q_j * Π_{k≠j}(1 - q_k) = q_j * exp(Σ log(1-q_k) - log(1-q_j))
            numer = q[Js] * np.exp(log_survive - log1mq[Js])
            denom = np.sum(numer)
            if denom > 0:
                resp[i, Js] = numer / denom

        # 更新（Beta(a0,b0)）
        new_q = np.zeros(J)
        for j in range(J):
            Nj_fail_resp = np.sum(resp[:, j])              
            Nj_total = np.sum(A[:, j])                    
            Nj_pass = Nj_total - np.sum((y == 1) & (A[:, j] == 1))
            
            num = (a0 - 1.0) + Nj_fail_resp
            den = (a0 + b0 - 2.0) + Nj_fail_resp + Nj_pass
            new_q[j] = np.clip(num / (den + 1e-12), 1e-6, 1 - 1e-6)

        # 收斂與 log-likelihood
        delta = np.linalg.norm(new_q - q)
        q = new_q
        # 當前 loglik
        loglik = 0.0
        for i in range(N):
            Js = idx_per_i[i]
            if Js.size == 0:
                pf = 0.0
            else:
                pf = 1 - np.exp(np.sum(np.log(1 - q[Js] + 1e-12)))
            pf = np.clip(pf, 1e-12, 1 - 1e-12)
            loglik += np.log(pf) if y[i] == 1 else np.log(1 - pf)
        loglik_hist.append(loglik)
        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"Iter {it:3d}  loglik={loglik:.3f}  delta={delta:.3e}")
        if delta < tol:
            break

    return q, resp, loglik_hist

def topk_overlap(true_vals, est_vals, k=3):
    it = np.argsort(-true_vals)[:k]
    ie = np.argsort(-est_vals)[:k]
    return len(set(it) & set(ie)) / k

def brier_score(y_true, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.mean((p - y_true)**2)

def plot_true_vs_est(q_true, q_est, title="True vs Estimated q_j"):
    plt.figure(figsize=(5, 4))
    plt.scatter(q_true, q_est, alpha=0.8)
    m = max(1.0, q_true.max()*1.05, q_est.max()*1.05)
    plt.plot([0, m], [0, m], 'r--')
    plt.xlim(0, m); plt.ylim(0, m)
    plt.xlabel("True q_j"); plt.ylabel("Estimated q_j")
    plt.title(title)
    plt.tight_layout(); plt.show()

def plot_responsibility_heatmap(resp, title="Responsibility heatmap"):
    fail_rows = np.where(resp.sum(axis=1) > 0)[0]
    if fail_rows.size == 0:
        print("No failures to plot.")
        return
    R = resp[fail_rows]
    order = np.argsort(-np.max(R, axis=1))
    R = R[order]
    plt.figure(figsize=(6, 4))
    plt.imshow(R, aspect='auto', cmap='viridis')
    plt.colorbar(label='responsibility')
    plt.xlabel("node j"); plt.ylabel("failed samples (sorted)")
    plt.title(title)
    plt.tight_layout(); plt.show()

def plot_counterfactual_influence(A, q_est, title="Counterfactual influence (approx)"):
    """
    近似反事實影響度：把某節點 j 修好(q_j=0)，平均失敗率下降多少
    """
    base_p = 1.0 - np.prod(1.0 - A * q_est, axis=1)     # (N,)
    N, J = A.shape
    delta = np.zeros(J)
    for j in range(J):
        q_abl = q_est.copy(); q_abl[j] = 0.0
        p_abl = 1.0 - np.prod(1.0 - A * q_abl, axis=1)
        delta[j] = np.mean(base_p - p_abl)
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(J), delta)
    plt.title(title); plt.xlabel("node j"); plt.ylabel("E[Δ p_fail]")
    plt.tight_layout(); plt.show()
    return delta

# （Poisson / Normal / Gamma）
def run_experiment(J=12, N=800, p_route=0.5, modes=("poisson", "normal", "gamma"),
                   a0=1.0, b0=8.0, seed=42, max_iter=200, tol=1e-6, plot=True):
    rng = np.random.default_rng(seed)
    for mode in modes:
        q_true = generate_q(J=J, mode=mode, seed=seed+1)
        A = rng.binomial(1, p_route, size=(N, J)).astype(float)
        y, p_fail_true = synthesize(A, q_true, rng)

        q_est, resp, hist = noisyor_em(A, y, a0=a0, b0=b0, max_iter=max_iter, tol=tol, verbose=False)

        # 評估
        corr = float(np.corrcoef(q_true, q_est)[0, 1])
        top3 = topk_overlap(q_true, q_est, k=3)
        # 用估計的 q 重新預測 p_fail，計算 Brier
        p_pred = 1.0 - np.prod(1.0 - A * q_est, axis=1)
        bs = brier_score(y, p_pred)

        print(f"[{mode}] corr={corr:.3f}  top3={top3:.2f}  Brier={bs:.3f}  mean(y)={y.mean():.3f}")

        if plot:
            plot_true_vs_est(q_true, q_est, title=f"{mode}  corr={corr:.2f}, top3={top3:.2f}")
            plot_responsibility_heatmap(resp, title=f"{mode}  responsibility on failed samples")
            _ = plot_counterfactual_influence(A, q_est, title=f"{mode}  counterfactual influence (approx)")


if __name__ == "__main__":
    run_experiment(
        J=12, N=800, p_route=0.5,
        modes=("poisson", "normal", "gamma"),
        a0=1.0, b0=8.0,          
        seed=42, max_iter=200,  
        tol=1e-6, plot=True
    )

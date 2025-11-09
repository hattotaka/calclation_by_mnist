import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import cvxpy as cp

# --- データ ---
X = np.array([
    [4, 2],
    [2, 4],
    [0, 0]
], dtype=float)
Y = np.array([4, 1, 1], dtype=float)

# λの範囲
lambdas = np.linspace(0, 50, 101)
p = 1  # Lp正則化パラメータ

# --- リッジ回帰 ---
def ridge_regression(X, Y, lam):
    XtX_lam = X.T @ X + lam * np.eye(X.shape[1])
    W_ridge = np.linalg.inv(XtX_lam) @ X.T @ Y
    Yhat_ridge = X @ W_ridge
    return W_ridge, Yhat_ridge

# --- Xλ構築（擬似逆使用） ---
def data_approximation(X, Y, lam):
    X_lambda = X + lam * np.linalg.pinv(X.T)
    XtX_lambda = X_lambda.T @ X_lambda
    W_Xlambda = np.linalg.inv(XtX_lambda) @ X_lambda.T @ Y
    Yhat_Xlambda = X @ W_Xlambda
    return X_lambda, W_Xlambda, Yhat_Xlambda

# --- Lp 正則化 ---
def lp_regression(X, Y, lam, p):
    beta = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.sum_squares(Y - X @ beta) + lam * cp.norm(beta, p)**p)
    prob = cp.Problem(objective)
    prob.solve()
    Yhat_lp = X @ beta.value
    return beta.value, Yhat_lp

# --- 特異値段階的底上げ ---
def svd_stabilize(X, n_stages):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_min, S_max = np.min(S), np.max(S)
    eps = (S_max - S_min) / (n_stages - 1)
    X_stages = []
    S_stages = []
    for stage in range(n_stages):
        threshold = S_min + stage * eps
        S_new = np.maximum(S, threshold)
        X_stages.append(U @ np.diag(S_new) @ VT)
        S_stages.append(S_new)
    return X_stages, np.array(S_stages), S

# --- 特異値段階的低下 ---
def svd_destabilize(X, n_stages):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_min, S_max = np.min(S), np.max(S)
    eps = (S_max - S_min) / (n_stages - 1)
    X_stages = []
    S_stages = []
    for stage in range(n_stages):
        threshold = S_max - stage * eps
        S_new = np.minimum(S, threshold)
        X_stages.append(U @ np.diag(S_new) @ VT)
        S_stages.append(S_new)
    return X_stages, np.array(S_stages), S

# --- SVD段階 ---
n_stages = 10
X_up_stages, S_up_stages, S_orig = svd_stabilize(X, n_stages)
X_down_stages, S_down_stages, _ = svd_destabilize(X, n_stages)

# --- 各段階での予測値と回帰係数 ---
def compute_stage_predictions(X_stages):
    Yhat_list = []
    W_list = []
    for X_stage in X_stages:
        W_stage = np.linalg.inv(X_stage.T @ X_stage) @ X_stage.T @ Y
        Yhat_stage = X @ W_stage
        Yhat_list.append(Yhat_stage[:2])
        W_list.append(W_stage)
    return np.array(Yhat_list), np.array(W_list)

Yhat_up_list, W_up_list = compute_stage_predictions(X_up_stages)
Yhat_down_list, W_down_list = compute_stage_predictions(X_down_stages)

# --- λループの通常正則化 ---
Yhat_ridge_list, Yhat_Xlambda_list, Yhat_lasso_list, Yhat_lp_list = [], [], [], []
W_ridge_list, W_lasso_list = [], []
X1_traj, X2_traj, corr_list = [], [], []

for lam in lambdas:
    # リッジ
    W_ridge, Yhat_ridge = ridge_regression(X, Y, lam)
    Yhat_ridge_list.append(Yhat_ridge[:2])
    W_ridge_list.append(W_ridge)

    # Xλ
    X_lambda, W_Xlambda, Yhat_Xlambda = data_approximation(X, Y, lam)
    Yhat_Xlambda_list.append(Yhat_Xlambda[:2])
    X1_traj.append(X_lambda[0, :2])
    X2_traj.append(X_lambda[1, :2])
    corr_list.append(np.corrcoef(X_lambda[:, 0], X_lambda[:, 1])[0, 1])

    # ラッソ
    lasso = Lasso(alpha=lam, fit_intercept=False)
    lasso.fit(X, Y)
    Yhat_lasso_list.append((X @ lasso.coef_)[:2])
    W_lasso_list.append(lasso.coef_)

    # Lp
    _, Yhat_lp = lp_regression(X, Y, lam, p)
    Yhat_lp_list.append(Yhat_lp[:2])

# --- numpy化 ---
Yhat_ridge_list = np.array(Yhat_ridge_list)
Yhat_Xlambda_list = np.array(Yhat_Xlambda_list)
Yhat_lasso_list = np.array(Yhat_lasso_list)
Yhat_lp_list = np.array(Yhat_lp_list)
W_ridge_list = np.array(W_ridge_list)
W_lasso_list = np.array(W_lasso_list)
X1_traj, X2_traj, corr_list = map(np.array, (X1_traj, X2_traj, corr_list))

# --- 図1: λごとの軌跡 + SVD段階 + X_stage自体の軌跡 ---
plt.figure(figsize=(10, 8))
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.grid(True)

# 元ベクトル
plt.quiver(0, 0, X[0, 0], X[1, 0], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.004, label='X1(Start)')
plt.quiver(0, 0, X[0, 1], X[1, 1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.004, label='X2(Start)')

# λ正則化軌跡
plt.plot(Yhat_ridge_list[:, 0], Yhat_ridge_list[:, 1], 'co', lw=2, label='Predict_Ridge')
# plt.plot(Yhat_Xlambda_list[:, 0], Yhat_Xlambda_list[:, 1], 'c-.', lw=2, label='Predict_Xλ')
# plt.plot(Yhat_lasso_list[:, 0], Yhat_lasso_list[:, 1], 'm:', lw=2, label='Predict_Lasso')
plt.plot(Yhat_lp_list[:, 0], Yhat_lp_list[:, 1], 'orange', lw=2, label='Predict_Lp')

# SVD底上げ段階の予測値
yhat_up_array = np.array(Yhat_up_list)  # shape=(n_stages, 3, 2)
plt.plot(yhat_up_array[:, 0], yhat_up_array[:, 1], 'ro', lw=2, label='Predict_SVD-Opt')

# X_stage 自体の軌跡（底上げ）
X_up_array = np.array(X_up_stages)  # shape=(n_stages, 3, 2)
plt.plot(X_up_array[:, 0, 0], X_up_array[:, 1, 0], 'b--', lw=2, label='SVD Optimize X1')
plt.plot(X_up_array[:, 0, 1], X_up_array[:, 1, 1], 'g--', lw=2, label='SVD Optimize X2')

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('predict + X_SVD')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# --- 図2: 特異値段階的変化 ---
plt.figure(figsize=(8, 4))
for i in range(S_up_stages.shape[1]):
    plt.plot(range(n_stages), S_up_stages[:, i], 'o-', label=f"σ{i+1}")
plt.xlabel('Stage')
plt.ylabel('SVD_Value')
plt.title('SVD-Up')
plt.legend()
plt.grid(True)
plt.show()

# --- 図3: X1, X2の角度変化とX1-X2のなす角（底上げSVD段階） ---
angles_X1, angles_X2, angles_between = [], [], []

for X_stage in X_up_stages:
    x1_vec = X_stage[:, 0]
    x2_vec = X_stage[:, 1]
    angles_X1.append(np.arctan2(x1_vec[1], x1_vec[0]))
    angles_X2.append(np.arctan2(x2_vec[1], x2_vec[0]))
    cos_theta = np.dot(x1_vec, x2_vec) / (np.linalg.norm(x1_vec) * np.linalg.norm(x2_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_between.append(np.arccos(cos_theta))

angles_X1 = np.degrees(angles_X1)
angles_X2 = np.degrees(angles_X2)
angles_between = np.degrees(angles_between)

plt.figure(figsize=(7, 4))
plt.plot(range(n_stages), angles_X1, 'o-', label='X1-angle')
plt.plot(range(n_stages), angles_X2, 's-', label='X2-angle')
plt.plot(range(n_stages), angles_between, 'd--', label='X1-X2-angle')
plt.xlabel('stage')
plt.ylabel('angle')
plt.title('X-angle')
plt.grid(True)
plt.legend()
plt.show()

# --- 図4: 回帰係数の軌跡（W1 vs W2） ---
plt.figure(figsize=(6, 6))
plt.plot(W_ridge_list[:, 0], W_ridge_list[:, 1], 'go-', label='Redge')
plt.plot(W_lasso_list[:, 0], W_lasso_list[:, 1], 'mo--', label='Lasso')
plt.plot(W_up_list[:, 0], W_up_list[:, 1], 'co-.', label='SVD-Opt')
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.xlabel('W1')
plt.ylabel('W2')
plt.title('Regression Coefficient')
plt.legend()
plt.grid(True)
plt.show()

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
Y = np.array([4, 0.5, 1], dtype=float)

# λの範囲
lambdas = np.linspace(0, 5, 11)

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

# --- Lp 正則化用関数 ---
def lp_regression(X, Y, lam, p):
    beta = cp.Variable(X.shape[1])
    objective = cp.Minimize(cp.sum_squares(Y - X @ beta) + lam * cp.norm(beta, p)**p)
    prob = cp.Problem(objective)
    prob.solve()
    Yhat_lp = X @ beta.value
    return beta.value, Yhat_lp

# --- 軌跡リスト ---
Yhat_ridge_list = []
Yhat_Xlambda_list = []
Yhat_lasso_list = []
Yhat_lp_list = []
X1_traj = []
X2_traj = []
corr_list = []

# Lp のパラメータ
p = 1

for lam in lambdas:
    # リッジ回帰
    W_ridge, Yhat_ridge = ridge_regression(X, Y, lam)
    Yhat_ridge_list.append(Yhat_ridge[:2])
    
    # データ近似
    X_lambda, W_Xlambda, Yhat_Xlambda = data_approximation(X, Y, lam)
    Yhat_Xlambda_list.append(Yhat_Xlambda[:2])
    
    # ラッソ回帰
    lasso = Lasso(alpha=lam, fit_intercept=False)
    lasso.fit(X, Y)
    Yhat_lasso = X @ lasso.coef_
    Yhat_lasso_list.append(Yhat_lasso[:2])
    
    # Lp 正則化回帰
    _, Yhat_lp = lp_regression(X, Y, lam, p)
    Yhat_lp_list.append(Yhat_lp[:2])
    
    # X1, X2 の軌跡（Xλ）
    X1_traj.append(X_lambda[0, :2])
    X2_traj.append(X_lambda[1, :2])
    
    # X1, X2 の相関係数（Xλ）
    x1 = X_lambda[:, 0]
    x2 = X_lambda[:, 1]
    corr = np.corrcoef(x1, x2)[0, 1]
    corr_list.append(corr)

# 配列化
Yhat_ridge_list = np.array(Yhat_ridge_list)
Yhat_Xlambda_list = np.array(Yhat_Xlambda_list)
Yhat_lasso_list = np.array(Yhat_lasso_list)
Yhat_lp_list = np.array(Yhat_lp_list)
X1_traj = np.array(X1_traj)
X2_traj = np.array(X2_traj)
corr_list = np.array(corr_list)

# --- 図1: 軌跡 + 予測値 ---
plt.figure(figsize=(8, 8))
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.grid(True)

# 元ベクトル
plt.quiver(0, 0, X[0, 0], X[1, 0], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.004, label='X1(Start)')
plt.quiver(0, 0, X[0, 1], X[1, 1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.004, label='X2(Start)')

# Xλ の軌跡
plt.plot(X1_traj[:, 0], X2_traj[:, 0], 'b--', lw=2, label='X1-Perturbation')
plt.plot(X1_traj[:, 1], X2_traj[:, 1], 'g--', lw=2, label='X2-Perturbation')

# 予測値の軌跡
plt.plot(Yhat_ridge_list[:, 0], Yhat_ridge_list[:, 1], 'co', lw=2.5, label='Predict-Redge')
plt.plot(Yhat_Xlambda_list[:, 0], Yhat_Xlambda_list[:, 1], 'r-.', lw=2, label='Predict-Xλ')
# plt.plot(Yhat_lasso_list[:, 0], Yhat_lasso_list[:, 1], 'm:', lw=2, label='ŷ(λ) ラッソ')
plt.plot(Yhat_lp_list[:, 0], Yhat_lp_list[:, 1], 'orange', lw=2, linestyle='-', marker='o', label='Predict-Lp')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('predict + X_λ')
plt.show()

# --- 図2: λ に応じた Xλ の相関係数 ---
plt.figure(figsize=(8, 5))
plt.plot(lambdas, corr_list, 'm-', lw=2, label='coeff')
plt.xlabel('λ')
plt.ylabel('coeff')
plt.title('coefficient X1-X2')
plt.grid(True)
plt.legend()
plt.show()

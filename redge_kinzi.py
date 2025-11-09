import numpy as np
import matplotlib.pyplot as plt

# --- データ ---
X = np.array([
    [4, 2],
    [2, 4],
    [0, 0]
], dtype=float)
Y = np.array([5, 1, 1], dtype=float)

# λの範囲
lambdas = np.linspace(0, 10, 11)

# --- リッジ回帰 ---
def ridge_regression(X, Y, lam):
    """リッジ回帰での予測値を計算"""
    XtX_lam = X.T @ X + lam * np.eye(X.shape[1])
    W_ridge = np.linalg.inv(XtX_lam) @ X.T @ Y
    Yhat_ridge = X @ W_ridge
    return W_ridge, Yhat_ridge

# --- Xλ構築（擬似逆使用） ---
def data_approximation(X, Y, lam, V=None, eigvals=None):
    """Xλ を擬似逆行列で構築し、Xλ 上で回帰して元空間での予測値を計算"""
    X_lambda = X + lam * np.linalg.pinv(X.T)
    XtX_lambda = X_lambda.T @ X_lambda
    W_Xlambda = np.linalg.inv(XtX_lambda) @ X_lambda.T @ Y
    Yhat_Xlambda = X @ W_Xlambda
    return X_lambda, W_Xlambda, Yhat_Xlambda

# --- X^+ の角度（列ベクトル間の相関） ---
X_pinv = np.linalg.pinv(X.T)
x1_plus = X_pinv[:, 0]
x2_plus = X_pinv[:, 1]
angle_Xplus = np.degrees(np.arccos(np.clip(np.dot(x1_plus, x2_plus) /
                                           (np.linalg.norm(x1_plus)*np.linalg.norm(x2_plus)), -1, 1)))
corr_Xplus = np.corrcoef(x1_plus, x2_plus)[0, 1]

# --- 軌跡・予測値・相関係数リスト ---
Yhat_ridge_list = []
Yhat_Xlambda_list = []
X1_traj = []
X2_traj = []
corr_list = []

for lam in lambdas:
    # リッジ回帰
    W_ridge, Yhat_ridge = ridge_regression(X, Y, lam)
    Yhat_ridge_list.append(Yhat_ridge[:2])
    
    # データ近似
    X_lambda, W_Xlambda, Yhat_Xlambda = data_approximation(X, Y, lam)
    Yhat_Xlambda_list.append(Yhat_Xlambda[:2])
    
    # X1, X2 の軌跡
    X1_traj.append(X_lambda[0, :2])
    X2_traj.append(X_lambda[1, :2])
    
    # X1, X2 の相関係数
    x1 = X_lambda[:, 0]
    x2 = X_lambda[:, 1]
    corr = np.corrcoef(x1, x2)[0, 1]
    corr_list.append(corr)

# 配列化
Yhat_ridge_list = np.array(Yhat_ridge_list)
Yhat_Xlambda_list = np.array(Yhat_Xlambda_list)
X1_traj = np.array(X1_traj)
X2_traj = np.array(X2_traj)
corr_list = np.array(corr_list)

# --- 図1: X1, X2 の軌跡 + 予測値 + X^+ の列ベクトル ---
plt.figure(figsize=(8, 8))
plt.axhline(0, color='gray', lw=1)
plt.axvline(0, color='gray', lw=1)
plt.grid(True)

# 元ベクトル X1, X2
plt.quiver(0, 0, X[0, 0], X[0, 1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.004, label='X1 (初期)')
plt.quiver(0, 0, X[1, 0], X[1, 1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.004, label='X2 (初期)')

# Xλ の軌跡
plt.plot(X1_traj[:, 0], X1_traj[:, 1], 'b--', lw=2, label='X1 λ軌跡')
plt.plot(X2_traj[:, 0], X2_traj[:, 1], 'r--', lw=2, label='X2 λ軌跡')

# リッジ予測値 ŷ(λ)
plt.plot(Yhat_ridge_list[:, 0], Yhat_ridge_list[:, 1], 'g-', lw=2.5, label='ŷ(λ) リッジ')

# Xλ回帰予測値 yXλ
plt.plot(Yhat_Xlambda_list[:, 0], Yhat_Xlambda_list[:, 1], 'c-.', lw=2, label='yXλ = X W_Xλ')

# X^+ の列ベクトル（λ→∞ の近似）
plt.quiver(0, 0, x1_plus[0], x1_plus[1], angles='xy', scale_units='xy', scale=1,
           color='orange', width=0.004, label='X^+ 列1 (λ→∞)')
plt.quiver(0, 0, x2_plus[0], x2_plus[1], angles='xy', scale_units='xy', scale=1,
           color='purple', width=0.004, label='X^+ 列2 (λ→∞)')

plt.title(f'X1, X2 の軌跡 + 予測値 + X^+ 列ベクトル\nX^+ の角度: {angle_Xplus:.1f}° 相関: {corr_Xplus:.2f}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# --- 図2: λ に応じた X1 と X2 の相関係数 ---
plt.figure(figsize=(8, 5))
plt.plot(lambdas, corr_list, 'm-', lw=2, label='Xλ 相関')
plt.axhline(corr_Xplus, color='orange', ls='--', lw=2, label='X^+ 相関 (λ→∞)')
plt.xlabel('λ')
plt.ylabel('X1, X2 相関係数')
plt.title('λ に応じた X1 と X2 の相関係数の推移')
plt.grid(True)
plt.legend()
plt.show()

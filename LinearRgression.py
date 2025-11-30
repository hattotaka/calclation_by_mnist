import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ===============================
# 座標系
# ===============================
def set_cords():
    n = np.array([1., 1., 1.]); n /= np.linalg.norm(n)
    u = np.array([-1., 1., 0.]); u /= np.linalg.norm(u)
    v = np.array([-1/2, -1/2, 1.]); v /= np.linalg.norm(v)
    o = np.array([1., 0., 0.])
    return np.column_stack([u, v, n]), o

def to_local_coords(points, R, o):
    points = np.atleast_2d(points)
    p_local = (R.T @ (points - o).T).T
    return p_local[:, :2] if p_local.shape[0] > 1 else p_local[0, :2]

# ===============================
# データ作成
# ===============================
def getData():
    np.random.seed(42)
    N = 100
    X1 = np.random.rand(N)
    X2 = np.random.rand(N)
    X3 = np.random.rand(N)
    beta_true1 = np.array([0.3, 0.5, 0.2])
    beta_true2 = np.array([0.2, 0.6, 0.7])
    beta0_true1 = 1.0
    beta0_true2 = 1.3
    Y1 = beta0_true1 + beta_true1[0]*X1 + beta_true1[1]*X2 + beta_true1[2]*X3 + 0.05*np.random.randn(N)
    Y2 = beta0_true2 + beta_true2[0]*X1 + beta_true2[1]*X2 + beta_true2[2]*X3 + 0.05*np.random.randn(N)
    df_X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    df_Y = pd.DataFrame({"Y1": Y1, "Y2": Y2})
    return df_X, df_Y

# ===============================
# 多出力回帰（β1+β2+β3=1 制約）
# ===============================
def LinearRegressionCoeffMulti(df_X, df_Y):
    Z1 = df_X["X1"] - df_X["X3"]
    Z2 = df_X["X2"] - df_X["X3"]
    Z = np.column_stack([Z1, Z2])
    target = df_Y.values - df_X["X3"].values[:, None]
    model = LinearRegression(fit_intercept=True)
    model.fit(Z, target)
    return model

# ===============================
# 実行
# ===============================
df_X, df_Y = getData()
model = LinearRegressionCoeffMulti(df_X, df_Y)

# 各出力の係数計算（β3は制約で計算）
coeffs = []
for i in range(df_Y.shape[1]):
    b1 = model.coef_[i,0]
    b2 = model.coef_[i,1]
    b3 = 1 - b1 - b2
    coeffs.append([b1,b2,b3])
coeffs = np.array(coeffs)  # shape (M,3)

# 平面上2D座標変換
R,o = set_cords()
coords_2d = to_local_coords(coeffs, R, o)  # shape (M,2)

# ===============================
# DataFrameにまとめる
# ===============================
df_result = pd.DataFrame(
    np.hstack([coeffs, coords_2d]),
    columns=["β1","β2","β3","u","v"],
    index=df_Y.columns
)
print(df_result)

# ===============================
# 三角形の頂点（元の3D座標から変換）
# ===============================
triangle_3d = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])
triangle_2d = to_local_coords(triangle_3d, R, o)
triangle_closed = np.vstack([triangle_2d, triangle_2d[0]])  # 閉じる

# ===============================
# 平面上にプロット
# ===============================
plt.figure(figsize=(6,6))

# 回帰結果の点
plt.scatter(df_result["u"], df_result["v"], color='blue', s=10, label="regression coefficient")

# 三角形を描画
plt.plot(triangle_closed[:,0], triangle_closed[:,1], 'g-', linewidth=1.)
triangle_labels = ["(1,0,0)", "(0,1,0)", "(0,0,1)"]
plt.text(triangle_2d[0,0]-0.11, triangle_2d[0,1]-0.02, 'LGP', color='black', fontsize=10)
plt.text(triangle_2d[1,0]+0.02, triangle_2d[1,1]-0.02, 'ICP', color='black', fontsize=10)
plt.text(triangle_2d[2,0]-0.03, triangle_2d[2,1]+0.02, 'IHP', color='black', fontsize=10)

plt.title("Distribution of regression coefficients")
plt.axis('equal')
plt.axis('off')
plt.legend()
plt.show()

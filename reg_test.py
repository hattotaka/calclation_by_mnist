import numpy as np
import matplotlib.pyplot as plt

# ===== 元データ =====
X1 = np.array([1.0, 2.0, 0.0])
X2 = np.array([2.0, 1.0, 0.0])
Y  = np.array([2.0, 0.5, 1.0])  # Z=1 とする

# ===== 回転行列（時計回りを正とする）=====
def R_clockwise(theta_deg):
    theta = np.deg2rad(theta_deg)
    Rz = np.array([
        [ np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta),  np.cos(theta), 0],
        [ 0,              0,             1]
    ])
    return Rz

# ===== 回帰関数 =====
def regress(Y, X1, X2):
    B = np.column_stack([X1, X2])
    beta = np.linalg.lstsq(B, Y, rcond=None)[0]
    Y_pred = B @ beta
    return beta, Y_pred

# ===== 角度計算 =====
def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))

# ===== 最適化 =====
def find_optimal_rotation(X1, X2, Y, p=1.05, search_range=90, step=1):
    beta_base, Y_pred_base = regress(Y, X1, X2)
    base_error = np.linalg.norm(Y - Y_pred_base)

    best_result = None
    best_angle = -np.inf

    for t1 in np.arange(-search_range, search_range+1, step):
        for t2 in np.arange(-search_range, search_range+1, step):
            X1_1 = R_clockwise(t1) @ X1
            X2_1 = R_clockwise(t2) @ X2

            beta_trans, Y_pred_plane = regress(Y, X1_1, X2_1)
            Y_back = beta_trans[0]*X1 + beta_trans[1]*X2

            err_back = np.linalg.norm(Y - Y_back)
            angle = angle_between(X1_1[:2], X2_1[:2])

            if err_back <= p * base_error and angle <= 90:
                if angle > best_angle:
                    best_angle = angle
                    best_result = (t1, t2, angle, err_back, beta_trans, Y_pred_plane, Y_back)

    return best_result, base_error

# ===== 軌跡の生成 =====
p_values = np.linspace(1.00, 1.10, 11)
trajectory = []

for p in p_values:
    best, base_err = find_optimal_rotation(X1, X2, Y, p=p)
    if best is not None:
        t1, t2, angle, err_back, beta_trans, Y_pred_plane, Y_back = best
        trajectory.append((p, Y_back[:2], err_back/base_err, t1, t2, angle, beta_trans))

# ===== 描画 =====
fig, ax = plt.subplots(figsize=(7,7))
origin = np.zeros(2)

# ベクトル描画
ax.quiver(*origin, *X1[:2], angles='xy', scale_units='xy', scale=1, color='blue', label='X1')
ax.quiver(*origin, *X2[:2], angles='xy', scale_units='xy', scale=1, color='green', label='X2')

# 軌跡描画
traj_points = np.array([t[1] for t in trajectory])
ax.plot(traj_points[:,0], traj_points[:,1], '-o', color='purple', label='β₁X₁+β₂X₂ 軌跡')

# 開始点と終了点を強調
ax.scatter(*traj_points[0], color='red', s=60, label=f'p={p_values[0]:.2f} (厳密)')
ax.scatter(*traj_points[-1], color='orange', s=60, label=f'p={p_values[-1]:.2f} (緩和)')

# 軸範囲の自動調整
all_vectors = np.vstack([X1[:2], X2[:2], traj_points, [[0,0]]])
x_min = np.min(all_vectors[:,0]) - 0.1
x_max = np.max(all_vectors[:,0]) + 0.1
y_min = np.min(all_vectors[:,1]) - 0.1
y_max = np.max(all_vectors[:,1]) + 0.1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.autoscale(enable=False)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

ax.set_title("誤差拡大率 p に対する β₁X₁+β₂X₂ の軌跡（Θ ∈ [-90°, 90°]）")
ax.legend()
plt.show()

# ===== 軌跡の数値出力 =====
print("p, Θ₁, Θ₂, angle(X₁₁,X₂₁), 誤差拡大率, β₁, β₂, β₁X₁+β₂X₂:")
for p, Yb, r, t1, t2, ang, beta in trajectory:
    print(f"p={p:.3f}, Θ₁={t1:5.1f}°, Θ₂={t2:5.1f}°, angle={ang:6.2f}°, err_ratio={r:6.4f}, "
          f"β₁={beta[0]:6.3f}, β₂={beta[1]:6.3f}, Y_back=({Yb[0]:6.3f},{Yb[1]:6.3f})")

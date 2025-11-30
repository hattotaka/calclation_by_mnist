import numpy as np
import matplotlib.pyplot as plt

# --- 1) 座標・平面基底 ---
A = np.array([1.,0.,0.])
B = np.array([0.,1.,0.])
C = np.array([0.,0.,1.])
P = np.array([0.5,0.2,0.3])
rx, ry, rz = 0.55, 0.30, 0.45

n = np.array([1.,1.,1.]) / np.sqrt(3.)
u = np.array([1., -1., 0.]); u /= np.linalg.norm(u)
v = np.cross(n, u); v /= np.linalg.norm(v)

# --- 2) 三角形 2D 座標 ---
sqrt2 = np.sqrt(2); sqrt6 = np.sqrt(6)
A2 = np.array([0.,0.]); B2 = np.array([sqrt2,0.]); C2 = np.array([sqrt2/2, sqrt6/2])
def to2d(coeff): return coeff[0]*A2 + coeff[1]*B2 + coeff[2]*C2
P2 = to2d(P)

# --- 3) 回転行列 ---
def rotation_matrix_xyz(theta_xy=0, theta_yz=0, theta_zx=0):
    cx, cy, cz = np.cos([theta_xy, theta_yz, theta_zx])
    sx, sy, sz = np.sin([theta_xy, theta_yz, theta_zx])
    
    Rxy = np.array([[ cx, -sx, 0],
                    [ sx,  cx, 0],
                    [  0,   0, 1]])
    
    Ryz = np.array([[1,  0,   0],
                    [0, cy, -sy],
                    [0, sy,  cy]])
    
    Rzx = np.array([[ cz, 0, sz],
                    [  0, 1,  0],
                    [-sz, 0, cz]])
    
    return Rzx @ Ryz @ Rxy

# 回転角度
theta_xy = np.radians(50)
theta_yz = np.radians(-10)
theta_zx = np.radians(20)
R = rotation_matrix_xyz(theta_xy, theta_yz, theta_zx)

# --- 4) 回転後の楕円体行列 ---
Ru_base = np.diag([1/rx**2, 1/ry**2, 1/rz**2])
Ru_rotated = R @ Ru_base @ R.T

# 2D投影行列
G = np.array([[u @ Ru_rotated @ u, u @ Ru_rotated @ v],
              [v @ Ru_rotated @ u, v @ Ru_rotated @ v]])
eigvals, eigvecs = np.linalg.eigh(G)
L_base = 1 / np.sqrt(eigvals)
E = eigvecs

# --- 5) TE 倍率リスト ---
TE_scales = [0.6, 0.8, 1.0, 1.2]
colors = plt.cm.Greens(np.linspace(0.5,1.0,len(TE_scales)))

# --- 6) 描画 ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect("equal"); ax.axis("off")

# 三角形塗りつぶし
tri = np.array([A2,B2,C2])
ax.fill(tri[:,0], tri[:,1], color="#FFE6BF", alpha=0.9, edgecolor="black", linewidth=1.2)

# 複数 TE 楕円
for i, (scale, color) in enumerate(zip(TE_scales, colors)):
    L = L_base * scale
    ellipse2d = []
    for t in np.linspace(0, 2*np.pi, 400):
        c = np.array([np.cos(t), np.sin(t)])
        alpha, beta = E @ (L * c)
        X = P + alpha*u + beta*v
        ellipse2d.append(to2d(X))
    ellipse2d = np.array(ellipse2d)
    ax.plot(ellipse2d[:,0], ellipse2d[:,1], color=color, linewidth=2)
    
    # 一番大きい楕円にラベル
    if i == len(TE_scales)-1:
        rightmost = ellipse2d[np.argmax(ellipse2d[:,0])]
        ax.text(rightmost[0]+0.03, rightmost[1], "TE Contour Lines",
                color='green', fontsize=12, fontstyle='italic')

# 目標ポートフォリオ
ax.scatter(P2[0], P2[1], c="black", s=110, zorder=5)
ax.text(P2[0]+0.03, P2[1]-0.02, "TargetWeight", fontsize=12, fontstyle='italic')

# ラベル
ax.text(A2[0]-0.11, A2[1]-0.02, "LGP", fontsize=14)
ax.text(B2[0]+0.02, B2[1]-0.02, "ICP", fontsize=14)
ax.text(C2[0]-0.03, C2[1]+0.02, "IHP", fontsize=14)

ax.set_title("2D Projection of Rotated TE Contours on x+y+z=1 Plane", fontsize=12)
plt.tight_layout()
plt.show()

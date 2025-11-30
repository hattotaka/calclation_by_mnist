import numpy as np
import matplotlib.pyplot as plt

# --- 1) 三角形頂点と基準点 ---
A = np.array([1.,0.,0.])
B = np.array([0.,1.,0.])
C = np.array([0.,0.,1.])
P = np.array([0.5,0.2,0.3])       # Target Weight
pat = 1
if pat == 0:
    Q = np.array([0.4,1.0,-0.4])      # Industry Index
elif pat == 1:
    Q = np.array([0.2,0.7,0.1])      # Industry Index

rx, ry, rz = 0.55, 0.30, 0.45     # 楕円体半径（TE 等高線）

# --- 2) 平面基底 ---
n = np.array([1.,1.,1.]) / np.sqrt(3.)
u = np.array([1., -1., 0.]); u /= np.linalg.norm(u)
v = np.cross(n, u); v /= np.linalg.norm(v)

# --- 3) 三角形 2D 座標 ---
sqrt2 = np.sqrt(2); sqrt6 = np.sqrt(6)
A2 = np.array([0.,0.]); B2 = np.array([sqrt2,0.]); C2 = np.array([sqrt2/2, sqrt6/2])
def to2d(coeff): return coeff[0]*A2 + coeff[1]*B2 + coeff[2]*C2
P2 = to2d(P); Q2 = to2d(Q)

# --- 4) 新しい三角形の中点 ---
midA = (A2 + Q2)/2; midB = (B2 + Q2)/2; midC = (C2 + Q2)/2
line_points = [midA, midB, midC, midA]

# --- 5) 回転行列 ---
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

# --- 6) 楕円体軸の回転 ---
theta_xy = np.radians(50)
theta_yz = np.radians(-10)
theta_zx = np.radians(20)
R = rotation_matrix_xyz(theta_xy, theta_yz, theta_zx)
Ru = np.diag([1/rx**2, 1/ry**2, 1/rz**2])
Ru_rotated = R @ Ru @ R.T

# --- 7) 2D投影用行列 ---
G = np.array([[u @ Ru_rotated @ u, u @ Ru_rotated @ v],
              [v @ Ru_rotated @ u, v @ Ru_rotated @ v]])
eigvals, eigvecs = np.linalg.eigh(G)
L = 1 / np.sqrt(eigvals)
E = eigvecs

# --- 8) 元の楕円（2D）生成 ---
num_pts = 400
ellipse2d_orig = []
for t in np.linspace(0, 2*np.pi, num_pts):
    c = np.array([np.cos(t), np.sin(t)])
    alpha, beta = E @ (L * c)
    X = P + alpha*u + beta*v
    ellipse2d_orig.append(to2d(X))
ellipse2d_orig = np.array(ellipse2d_orig)

# --- 9) 距離計算関数 ---
def distance_point_to_segment(P0, L1, L2):
    v = L2-L1; w = P0-L1
    c1 = np.dot(w,v); c2 = np.dot(v,v)
    b = 0 if c2==0 else np.clip(c1/c2,0,1)
    Pb = L1 + b*v
    return np.linalg.norm(P0-Pb), Pb

# --- 10) 楕円を拡大して最初に接する点のみ取得 ---
scale = 0.3
scale_step = 0.001
max_scale = 5.0
tolerance = 0.001
ellipse2d_scaled = ellipse2d_orig.copy()

first_contact_pt = None
first_contact_ellipse_pt = None

while scale < max_scale:
    scaled = P2 + scale*(ellipse2d_orig - P2)
    contact_found = False
    for pt in scaled:
        for j in range(len(line_points)-1):
            d, Pb = distance_point_to_segment(pt, line_points[j], line_points[j+1])
            if d <= tolerance:
                first_contact_pt = Pb
                first_contact_ellipse_pt = pt
                contact_found = True
                break
        if contact_found:
            break
    if contact_found:
        ellipse2d_scaled = scaled
        break
    scale += scale_step
else:
    ellipse2d_scaled = P2 + scale*(ellipse2d_orig - P2)

# 接点を1点だけ格納
contact_points = np.array([first_contact_pt])
ellipse_contact_points = np.array([first_contact_ellipse_pt])
orig_points = np.array([2*first_contact_pt - Q2])

# --- 11) 描画 ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect("equal"); ax.axis("off")

# 元の三角形
tri = np.array([A2,B2,C2])
ax.fill(tri[:,0], tri[:,1], color="#FFE6BF", alpha=0.9, edgecolor="black", linewidth=1.2)

# Target Weight
ax.scatter(P2[0], P2[1], c="black", s=110, zorder=5)
ax.text(P2[0]+0.03,P2[1]-0.02,"Target Weight", fontsize=12, fontstyle='italic')

# Industry Index
ax.scatter(Q2[0], Q2[1], c="red", s=110, zorder=6)
ax.text(Q2[0]+0.03,Q2[1]-0.02,"Industry Index", fontsize=12, color="red", fontstyle='italic')

# 新しい三角形（中点）
new_tri = np.array([midA, midB, midC, midA])
ax.plot(new_tri[:,0], new_tri[:,1], linestyle='--', color='blue', linewidth=1.5)

# 元の頂点→Q2 線
for vertex in [A2,B2,C2]:
    ax.plot([vertex[0], Q2[0]], [vertex[1], Q2[1]], linestyle='--', color='gray', linewidth=1.2)

# TE楕円
ax.plot(ellipse2d_scaled[:,0], ellipse2d_scaled[:,1], color="green", linewidth=2, label="TE Ellipse")

# 接点（最初に接した1点のみ）
ax.scatter(ellipse_contact_points[:,0], ellipse_contact_points[:,1],
           color="purple", s=100, label="Contact Point")
ax.text(ellipse_contact_points[0,0]+0.04, ellipse_contact_points[0,1]+0.01,
        "Overall Portfolio", color="purple", fontsize=12, fontstyle='italic')

# 対応する元の三角形の点
ax.scatter(orig_points[:,0], orig_points[:,1],
           color="orange", s=100, label="Original Point")
ax.text(orig_points[0,0]+0.04, orig_points[0,1]+0.00,
        "Supplement Portforio", color="orange", fontsize=12, fontstyle='italic')

# 接点から元の三角形の点への矢印（細い点線）
for e_pt, o_pt in zip(ellipse_contact_points, orig_points):
    ax.annotate("", xy=o_pt, xytext=e_pt,
                arrowprops=dict(arrowstyle='->', linestyle='dotted', color='black', linewidth=1))

# 元の三角形ラベル
ax.text(A2[0]-0.11,A2[1]-0.02,"LGP", fontsize=14)
ax.text(B2[0]+0.02,B2[1]-0.02,"ICP", fontsize=14)
ax.text(C2[0]-0.03,C2[1]+0.02,"IHP", fontsize=14)

ax.set_title("2D Projection with First TE Ellipse Contact and Labeled Dotted Arrow", fontsize=12)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- 1) 三角形頂点と基準点 ---
A = np.array([1.,0.,0.])
B = np.array([0.,1.,0.])
C = np.array([0.,0.,1.])
pat = 0
if pat == 0:
    Q = np.array([0.4,1.0,-0.4])      # Industry Index
elif pat == 1:
    Q = np.array([0.2,0.7,0.1])      # Industry Index
P = np.array([0.5,0.2,0.3])       # Target Weight

# --- 2) 三角形 2D 座標 ---
sqrt2 = np.sqrt(2); sqrt6 = np.sqrt(6)
A2 = np.array([0.,0.]); B2 = np.array([sqrt2,0.]); C2 = np.array([sqrt2/2, sqrt6/2])
def to2d(coeff): return coeff[0]*A2 + coeff[1]*B2 + coeff[2]*C2
P2 = to2d(P); Q2 = to2d(Q)

# --- 3) n:1 比率 ---
n = 6.0  # n:1 の比率
def interpolate_n1(vertex, Q2, n):
    # vertex と Q2 の n:1 線形補間
    return (n*vertex + Q2)/(n+1)

# 点線三角形の頂点
midA = interpolate_n1(A2, Q2, n)
midB = interpolate_n1(B2, Q2, n)
midC = interpolate_n1(C2, Q2, n)
line_points = [midA, midB, midC, midA]  # 点線三角形閉じる

# --- 4) TargetWeight に対応する元の三角形の点 ---
# P2 = (n*X + Q2)/(n+1) → X = ((n+1)*P2 - Q2)/n
mapped_point = ((n+1)*P2 - Q2)/n

# --- 5) 描画 ---
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect("equal"); ax.axis("off")

# 元の三角形
tri = np.array([A2,B2,C2])
ax.fill(tri[:,0], tri[:,1], color="#FFE6BF", alpha=0.9, edgecolor="black", linewidth=1.2)

# Industry Index
ax.scatter(Q2[0], Q2[1], c="red", s=110, zorder=6)
ax.text(Q2[0]+0.03,Q2[1]-0.02,"Industry Index", fontsize=12, color="red", fontstyle='italic')

# Target Weight
ax.scatter(P2[0], P2[1], c="black", s=110, zorder=5)
ax.text(P2[0]+0.03,P2[1]-0.02,"Target Weight", fontsize=12, fontstyle='italic')

# 点線三角形（元頂点と Q2 の n:1 線形補間）
new_tri = np.array([midA, midB, midC, midA])
ax.plot(new_tri[:,0], new_tri[:,1], linestyle='--', color='blue', linewidth=1.5)

# 元の頂点→Q2 線
for vertex in [A2,B2,C2]:
    ax.plot([vertex[0], Q2[0]], [vertex[1], Q2[1]], linestyle='--', color='gray', linewidth=1.2)
    
# Target Weight に対応する元の三角形上の点（Supplement Portfolio）
ax.scatter(mapped_point[0], mapped_point[1], c="orange", s=100, label="Supplement Portforio")
ax.text(mapped_point[0]+0.03, mapped_point[1]+0.01,
        "Supplement Portforio", color="orange", fontsize=12, fontstyle='italic')

# 矢印を Target Weight → Supplement Portfolio
ax.annotate("", xy=mapped_point, xytext=P2,
            arrowprops=dict(arrowstyle='->', linestyle='dotted', color='black', linewidth=1))

# 元の三角形ラベル
ax.text(A2[0]-0.11,A2[1]-0.02,"LGP", fontsize=14)
ax.text(B2[0]+0.02,B2[1]-0.02,"ICP", fontsize=14)
ax.text(C2[0]-0.03,C2[1]+0.02,"IHP", fontsize=14)

ax.set_title("TargetWeight Mapped to Original Triangle with n:1 Ratio", fontsize=12)
plt.tight_layout()
plt.show()

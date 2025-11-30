import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
import matplotlib.cm as cm

# ───────────────────────────────
# 1) 基本設定
# ───────────────────────────────
AXIS_LIMIT_MIN = -0.05
AXIS_LIMIT_MAX = 1.05

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")

# ───────────────────────────────
# 2) 背景・枠・目盛り軸・スパイン・グリッド削除
# ───────────────────────────────
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
ax.grid(False)
for axis in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    axis.set_visible(False)
ax.xaxis.line.set_color('white')
ax.yaxis.line.set_color('white')
ax.zaxis.line.set_color('white')

# ───────────────────────────────
# 3) 軸 (細グレー + 矢印)
# ───────────────────────────────
arrow_props = dict(mutation_scale=22, arrowstyle='-|>', color="gray", lw=1.2)

# X方
ax.quiver(0, 0, 0, 1.2, 0, 0, color="gray", arrow_length_ratio=0.07, linewidth=1.2)
# Y方向
ax.quiver(0, 0, 0, 0, 1.2, 0, color="gray", arrow_length_ratio=0.07, linewidth=1.2)
# Z方向
ax.quiver(0, 0, 0, 0, 0, 1.2, color="gray", arrow_length_ratio=0.07, linewidth=1.2)

# ラベル
ax.text(1.15, 0.00, 0.11, "$LGP$", fontsize=20)
ax.text(0.00, 1.02, 0.03, "$ICP$", fontsize=20)
ax.text(0.00, 0.02, 1.04, "$IHP$", fontsize=20)


# ───────────────────────────────
# 4) (x + y + z = 1) 三角平面
# ───────────────────────────────
triangle = np.array([[1,0,0], [0,1,0], [0,0,1]])
poly = Poly3DCollection([triangle], alpha=0.52, facecolor="#FFD8A8", edgecolor="navy", linewidth=1.6)
ax.add_collection3d(poly)


# ───────────────────────────────
# 5) プロット点
# ───────────────────────────────
plot_point = (0.5, 0.2, 0.3)
offset = 0.3
point_color = (1.0, 0.2, 0.2, 0.2)
ax.scatter(plot_point[0]+offset, plot_point[1]+offset, plot_point[2]+offset, s=110, c=[point_color], edgecolors="black", linewidth=0.8)


# ───────────────────────────────
# 6) 楕円体（透過＋陰影）
# ───────────────────────────────
cx, cy, cz = plot_point
rx, ry, rz = 0.55, 0.30, 0.45

u = np.linspace(0, 2*np.pi, 120)
v = np.linspace(0, np.pi, 120)
x = rx * np.outer(np.cos(u), np.sin(v)) + cx
y = ry * np.outer(np.sin(u), np.sin(v)) + cy
z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz

light = LightSource(azdeg=45, altdeg=45)
rgb = light.shade(z, cmap=cm.viridis, vert_exag=1.0, blend_mode='soft')

# ↙ ここで透明度を追加（0.45 でほどよい透過）
rgba = np.zeros(rgb.shape)
rgba[..., :3] = rgb[..., :3]
rgba[..., 3] = 0.35

ax.plot_surface(
    x, y, z,
    rstride=1, cstride=1,
    facecolors=rgba,
    edgecolor="black", linewidth=0.10,
    shade=False
)

# ───────────────────────────────
# 7) カメラアングル・描画範囲
# ───────────────────────────────
ax.view_init(elev=28, azim=42)
ax.set_xlim(AXIS_LIMIT_MIN, AXIS_LIMIT_MAX)
ax.set_ylim(AXIS_LIMIT_MIN, AXIS_LIMIT_MAX)
ax.set_zlim(AXIS_LIMIT_MIN, AXIS_LIMIT_MAX)

plt.tight_layout()
plt.show()

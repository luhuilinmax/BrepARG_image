import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

debug_pkl = "/root/autodl-tmp/AR/BrepARG_m/result/debug/abc_single_1776005862_820015_debug.pkl"
out_png = "/root/autodl-tmp/AR/BrepARG_m/result/debug/abc_single_1776005862_820015_geom_vis.png"

data = pickle.load(open(debug_pkl, "rb"))
joint = data.get("joint_opt", {})

edge_wcs = np.array(joint.get("edge_wcs", []), dtype=float)      # (E,32,3)
surf_wcs = np.array(joint.get("surf_wcs", []), dtype=float)      # (F,32,32,3)
unique_vertices = np.array(joint.get("unique_vertices", []), dtype=float)  # (V,3)

print("edge_wcs shape:", edge_wcs.shape)
print("surf_wcs shape:", surf_wcs.shape)
print("unique_vertices shape:", unique_vertices.shape)

fig = plt.figure(figsize=(18, 6))

# 1) edges
ax1 = fig.add_subplot(131, projection="3d")
if edge_wcs.ndim == 3:
    for i, e in enumerate(edge_wcs):
        # 退化边（唯一点少）用红色
        uniq = len(np.unique(np.round(e, 6), axis=0))
        c = "red" if uniq < 4 else "steelblue"
        ax1.plot(e[:,0], e[:,1], e[:,2], c=c, linewidth=1)
ax1.set_title("edge_wcs (red = degenerate)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# 2) surfaces (sparse points)
ax2 = fig.add_subplot(132, projection="3d")
if surf_wcs.ndim == 4:
    # 每个面抽稀，避免太密
    for s in surf_wcs:
        pts = s[::4, ::4, :].reshape(-1, 3)
        ax2.scatter(pts[:,0], pts[:,1], pts[:,2], s=2, alpha=0.35)
ax2.set_title("surf_wcs sparse points")
ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

# 3) overlay edge + vertices
ax3 = fig.add_subplot(133, projection="3d")
if edge_wcs.ndim == 3:
    for e in edge_wcs:
        ax3.plot(e[:,0], e[:,1], e[:,2], c="gray", linewidth=0.8, alpha=0.8)
if unique_vertices.ndim == 2 and len(unique_vertices) > 0:
    ax3.scatter(unique_vertices[:,0], unique_vertices[:,1], unique_vertices[:,2],
                c="orange", s=60, depthshade=True, label="unique_vertices")
ax3.set_title(f"overlay (unique_vertices={len(unique_vertices)})")
ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
if len(unique_vertices) > 0:
    ax3.legend(loc="upper right")

plt.tight_layout()
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=200)
print("saved:", out_png)
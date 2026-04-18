import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

debug_file = Path("/root/autodl-tmp/AR/BrepARG_m/result/debug/abc_multi_single_*.pkl")
matches = sorted(debug_file.parent.glob(debug_file.name))
assert matches, "没有找到 debug pkl"
p = matches[-1]
print("using:", p)

obj = pickle.load(open(p, "rb"))
cad = obj.get("cad_data", {})
joint = obj.get("joint_opt", {})

surf_ncs = np.array(cad.get("surf_ncs", []), dtype=float)      # 重建前
edge_ncs = np.array(cad.get("edge_ncs", []), dtype=float)      # 重建前
surf_wcs = np.array(joint.get("surf_wcs", []), dtype=float)    # joint_opt后
edge_wcs = np.array(joint.get("edge_wcs", []), dtype=float)    # joint_opt后
verts = np.array(joint.get("unique_vertices", []), dtype=float)

fig = plt.figure(figsize=(16, 10))

# A: 重建前 edge_ncs
ax1 = fig.add_subplot(221, projection="3d")
if edge_ncs.ndim == 3:
    for e in edge_ncs:
        ax1.plot(e[:,0], e[:,1], e[:,2], c="steelblue", linewidth=0.8)
ax1.set_title("Pre-recon edge_ncs")

# B: 重建前 surf_ncs 稀疏点
ax2 = fig.add_subplot(222, projection="3d")
if surf_ncs.ndim == 4:
    for s in surf_ncs:
        pts = s[::4, ::4, :].reshape(-1, 3)
        ax2.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, alpha=0.25)
ax2.set_title("Pre-recon surf_ncs (sparse)")

# C: joint_opt 后 edge_wcs，退化边标红
ax3 = fig.add_subplot(223, projection="3d")
if edge_wcs.ndim == 3:
    for e in edge_wcs:
        uniq = len(np.unique(np.round(e, 6), axis=0))
        c = "red" if uniq < 4 else "gray"
        ax3.plot(e[:,0], e[:,1], e[:,2], c=c, linewidth=0.9)
ax3.set_title("Post-joint edge_wcs (red=degenerate)")

# D: joint_opt 后 surf_wcs + unique_vertices
ax4 = fig.add_subplot(224, projection="3d")
if surf_wcs.ndim == 4:
    for s in surf_wcs:
        pts = s[::4, ::4, :].reshape(-1, 3)
        ax4.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, alpha=0.15)
if verts.ndim == 2 and len(verts) > 0:
    ax4.scatter(verts[:,0], verts[:,1], verts[:,2], c="orange", s=40, label=f"verts={len(verts)}")
    ax4.legend()
ax4.set_title("Post-joint surf_wcs + unique_vertices")

out = p.with_name(p.stem + "_vis4.png")
plt.tight_layout()
plt.savefig(out, dpi=220)
print("saved:", out)
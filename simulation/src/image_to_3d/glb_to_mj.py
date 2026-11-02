"""image→3D 모델(TripoSG, SF3D)이 만든 GLB 를 씬에서 쓰는 형식(OBJ, cm 단위, Z-up, 바닥 중심 원점) + PNG 텍스처로 바꾼다.

  python src/image_to_3d/glb_to_mj.py third_party/triposg_out/pill_bottle.glb pill_bottle 8.5
    -> models/mj/pill_bottle.obj (+ 텍스처가 있으면 models/textures/pill_bottle.png)  (높이 8.5 cm 로 맞춤)
  simulation/ 에서 실행. trimesh 만 있으면 된다 (.venv-triposg 에 들어 있음).
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj

src, name, height_cm = sys.argv[1], sys.argv[2], float(sys.argv[3])
models = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("models")

mesh = trimesh.load(src, force="mesh")

# 세우기: 모델마다 출력 축이 다르고(SF3D 는 Y-up + 카메라 각도만큼 기울어짐, TripoSG 는 옆으로 누운 채로 나오기도 함),
# 그래서 축을 가정하지 않고 실제로 바닥에 닿는 면을 찾는다: convex hull 의 평면 중 mesh 면이 가장 넓게 붙어 있는 평면이 바닥.
# (컵 입구처럼 테두리만 닿는 평면은 붙은 면적이 작아서 걸러진다)
hull = mesh.convex_hull
best_area, bottom = -1.0, None
for facet, normal in zip(hull.facets, hull.facets_normal):
    offset = normal @ hull.vertices[hull.faces[facet[0], 0]]
    on_plane = (mesh.face_normals @ normal > 0.9) & (np.abs(mesh.triangles_center @ normal - offset) < 0.01 * mesh.scale)
    if mesh.area_faces[on_plane].sum() > best_area:
        best_area, bottom = mesh.area_faces[on_plane].sum(), normal
mesh.apply_transform(trimesh.geometry.align_vectors(bottom, [0, 0, -1]))

# 실측 높이(cm)에 맞춰 등방 스케일. 씬 XML 에서 mesh scale="0.01 0.01 0.01" 로 m 변환
mesh.apply_scale(height_cm / mesh.extents[2])

# 바닥면 중심을 원점으로 (body pos 의 z 를 상판 높이로 두면 그대로 올라감)
lo, hi = mesh.bounds
mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

text, files = export_obj(mesh, include_texture=True, return_texture=True, mtl_name=f"{name}.mtl")
(models / "mj" / f"{name}.obj").write_text(text)
pngs = [data for fname, data in files.items() if fname.lower().endswith(".png")]
if pngs:
    (models / "textures" / f"{name}.png").write_bytes(pngs[0])

print(f"{name}: {mesh.extents.round(2)} cm (x y z), {len(mesh.vertices)} verts, texture={'yes' if pngs else 'no'}")

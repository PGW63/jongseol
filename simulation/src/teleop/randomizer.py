"""models/automatic.json 에 적힌 범위 안에서 씬의 물체 위치와 바닥·조명을 흩뜨린다.

물체마다 위에서 내려다본 윤곽(메쉬 꼭짓점의 볼록 껍질)을 만들어 두고,
옮긴 윤곽이 트롤리 밖으로 나가거나 다른 것과 겹치면 다시 뽑는다.
"""

import json
from pathlib import Path

import mujoco
import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    """2D 점들의 볼록 껍질 (반시계 방향, monotone chain)."""
    pts = sorted(set(map(tuple, np.round(points, 6))))
    if len(pts) <= 2:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(np.subtract(out[-1], out[-2]), np.subtract(p, out[-2])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(reversed(pts)))


def separated(a: np.ndarray, b: np.ndarray, margin: float) -> bool:
    """볼록 다각형 a, b 가 margin 이상 떨어져 있으면 True (분리축 검사)."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        with np.errstate(all="ignore"):  # Apple Silicon numpy 가 matmul 에 내는 가짜 경고
            pa, pb = a @ normals.T, b @ normals.T
        gap = np.maximum(pa.min(0) - pb.max(0), pb.min(0) - pa.max(0))
        if gap.max() >= margin:
            return True
    return False


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class SceneRandomizer:
    def __init__(self, model: mujoco.MjModel, path: Path, seed: int | None = None):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.lo = np.array([spec["bounds"]["x"][0], spec["bounds"]["y"][0]])
        self.hi = np.array([spec["bounds"]["x"][1], spec["bounds"]["y"][1]])
        self.margin = float(spec.get("margin", 0.0))
        self.max_tries = int(spec.get("max_tries", 200))

        # 원래 자리: 첫 키프레임에서의 월드 자세
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        # 바닥·조명: 원래 값을 기억해 두고 매번 거기서부터 흔든다
        self.floor = spec.get("floor")
        self.light = spec.get("light")
        if self.floor:
            self.floor_geom = model.geom(self.floor["geom"]).id
            self.floor_mats = [model.material(n).id for n in self.floor["materials"]]
        self.nominal = {
            "geom_matid": model.geom_matid.copy(),
            "mat_rgba": model.mat_rgba.copy(),
            "mat_texrepeat": model.mat_texrepeat.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "head_diffuse": model.vis.headlight.diffuse.copy(),
            "head_ambient": model.vis.headlight.ambient.copy(),
        }

        self.obstacles = [self._footprint(data, [model.body(n).id]) for n in spec.get("obstacles", [])]
        self.objects = []
        for name, o in spec["objects"].items():
            ids = [model.body(n).id for n in o["bodies"]]
            hull = self._footprint(data, [b for b in range(model.nbody) if self._root_in(b, ids)])
            bodies = []
            for b in ids:
                free = model.body_jntnum[b] == 1 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
                if not free:
                    # 고정 물체는 body_pos 를 월드 좌표로 고쳐 쓰므로, 부모가 원점에 그대로 있어야 한다
                    parent = model.body_parentid[b]
                    if not (np.allclose(data.xpos[parent], 0) and np.allclose(data.xquat[parent], [1, 0, 0, 0])):
                        raise ValueError(f"{model.body(b).name}: 부모 body 가 원점에 있지 않아 옮길 수 없습니다")
                bodies.append({
                    "id": b,
                    "qadr": int(model.jnt_qposadr[model.body_jntadr[b]]) if free else None,
                    "pos": data.xpos[b].copy(),
                    "quat": data.xquat[b].copy(),
                })
            self.objects.append({
                "name": name,
                "bodies": bodies,
                "hull": hull,
                "center": (hull.min(0) + hull.max(0)) / 2,
                "x": o["x"], "y": o["y"], "rot": np.radians(o["rot_deg"]),
            })

    def _root_in(self, body: int, roots: list[int]) -> bool:
        while body != 0:
            if body in roots:
                return True
            body = self.model.body_parentid[body]
        return False

    def _footprint(self, data: mujoco.MjData, bodies: list[int]) -> np.ndarray:
        m, pts = self.model, []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in bodies or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            i = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
            with np.errstate(all="ignore"):  # Apple Silicon numpy 가 matmul 에 내는 가짜 경고
                pts.append((verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, :2])
        if not pts:
            raise ValueError(f"윤곽을 만들 메쉬가 없습니다: {[m.body(b).name for b in bodies]}")
        return convex_hull(np.vstack(pts))

    def sample(self) -> dict[str, tuple[float, float, float]] | None:
        """모든 물체의 (dx, dy, 회전 rad) 를 뽑는다. 겹치지 않는 배치를 못 찾으면 None."""
        placed = list(self.obstacles)
        result = {}
        for o in self.objects:
            for _ in range(self.max_tries):
                d = np.array([self.rng.uniform(*o["x"]), self.rng.uniform(*o["y"])])
                theta = self.rng.uniform(*o["rot"])
                hull = (o["hull"] - o["center"]) @ _rot(theta).T + o["center"] + d
                inside = (hull >= self.lo).all() and (hull <= self.hi).all()
                if inside and all(separated(hull, other, self.margin) for other in placed):
                    placed.append(hull)
                    result[o["name"]] = (float(d[0]), float(d[1]), float(theta))
                    break
            else:
                return None
        return result

    def apply(self, data: mujoco.MjData) -> dict[str, tuple[float, float, float]] | None:
        """새 배치를 뽑아 씬에 적용한다. 못 찾으면 원래 자리로 두고 None 을 돌려준다.

        키프레임으로 되돌린 직후에 부른다. 부른 쪽에서 mj_forward 를 해야 한다.
        """
        offsets = None
        for _ in range(20):  # 먼저 놓은 물체 때문에 뒤 물체 자리가 없으면 처음부터 다시
            offsets = self.sample()
            if offsets is not None:
                break
        self._place(data, offsets)
        self._appearance(randomize=True)
        return offsets

    def restore(self, data: mujoco.MjData) -> None:
        """모든 물체와 바닥·조명을 원래대로 둔다."""
        self._place(data, None)
        self._appearance(randomize=False)

    def _appearance(self, randomize: bool) -> None:
        """바닥 재질과 조명 밝기. 모델 값을 고치므로 다음 렌더부터 바로 보인다."""
        m, n = self.model, self.nominal
        m.geom_matid[:] = n["geom_matid"]
        m.mat_rgba[:] = n["mat_rgba"]
        m.mat_texrepeat[:] = n["mat_texrepeat"]
        m.light_diffuse[:] = n["light_diffuse"]
        m.vis.headlight.diffuse[:] = n["head_diffuse"]
        m.vis.headlight.ambient[:] = n["head_ambient"]
        if not randomize:
            return
        if self.floor:
            mat = int(self.rng.choice(self.floor_mats))
            m.geom_matid[self.floor_geom] = mat
            tint = self.floor.get("tint", 0.0)
            scale = self.rng.uniform(*self.floor.get("brightness", [1.0, 1.0])) * (1 + self.rng.uniform(-tint, tint, 3))
            m.mat_rgba[mat, :3] = np.clip(n["mat_rgba"][mat, :3] * scale, 0, 1)
            if "texrepeat" in self.floor:
                # 평균 반복 횟수만 바꾸고 가로:세로 비는 재질에 적힌 대로 둔다 (사진이 늘어나지 않게)
                ratio = n["mat_texrepeat"][mat] / n["mat_texrepeat"][mat].mean()
                m.mat_texrepeat[mat] = ratio * self.rng.uniform(*self.floor["texrepeat"])
        if self.light:
            k = self.rng.uniform(*self.light["brightness"])
            m.light_diffuse[:] = np.clip(n["light_diffuse"] * k, 0, 1)
            m.vis.headlight.diffuse[:] = np.clip(n["head_diffuse"] * k, 0, 1)
            m.vis.headlight.ambient[:] = np.clip(n["head_ambient"] * k, 0, 1)

    def _place(self, data: mujoco.MjData, offsets: dict[str, tuple[float, float, float]] | None) -> None:
        for o in self.objects:
            dx, dy, theta = offsets[o["name"]] if offsets else (0.0, 0.0, 0.0)
            qz = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])
            for b in o["bodies"]:
                pos = b["pos"].copy()
                pos[:2] = (b["pos"][:2] - o["center"]) @ _rot(theta).T + o["center"] + [dx, dy]
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, qz, b["quat"])
                if b["qadr"] is None:
                    self.model.body_pos[b["id"]] = pos
                    self.model.body_quat[b["id"]] = quat
                else:
                    data.qpos[b["qadr"]:b["qadr"] + 3] = pos
                    data.qpos[b["qadr"] + 3:b["qadr"] + 7] = quat

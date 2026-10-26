"""리더암으로 MuJoCo 씬의 SO-101 을 움직이는 공용 루프."""

import time

import mujoco
import numpy as np

from teleop.constants import JOINT_NAMES
from teleop.leader_mj.config import SO101LeaderMjConfig
from teleop.leader_mj.teleop_so101_leader_mj import SO101LeaderMj
from teleop.mapping import rad_to_normalized
from teleop.randomizer import SceneRandomizer


class LeaderSim:
    """씬을 열고, 리더암을 읽어 한 틱(1/fps 초)씩 시뮬레이션을 진행한다."""

    def __init__(self, cfg: SO101LeaderMjConfig):
        self.cfg = cfg
        # 절대경로로 연다: 씬 안의 <model file="../so101/so101.xml"> 이 상대경로에서는 안 열릴 수 있다
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path.resolve()))
        self.data = mujoco.MjData(self.model)
        self.act_ids = [self.model.actuator(cfg.joint_prefix + j).id for j in JOINT_NAMES]
        self.qpos_adr = [int(self.model.joint(cfg.joint_prefix + j).qposadr[0]) for j in JOINT_NAMES]
        self.qvel_adr = [int(self.model.joint(cfg.joint_prefix + j).dofadr[0]) for j in JOINT_NAMES]
        self.steps_per_tick = max(1, round(1.0 / cfg.fps / self.model.opt.timestep))
        self.randomizer = (
            SceneRandomizer(self.model, cfg.randomize_path.resolve(), cfg.randomize_seed)
            if cfg.randomize_path else None
        )
        self.teleop = SO101LeaderMj(cfg)
        self.read_ms = 0.0

    def connect(self) -> None:
        # calibrate=False: 캘리브레이션이 안 맞아도 입력을 기다리며 멈추지 않게
        self.teleop.connect(calibrate=False)
        if not self.teleop.is_calibrated:
            print("경고: 모터에 저장된 캘리브레이션이 파일과 다릅니다. lerobot-calibrate 를 다시 실행하세요.")
        self.reset()

    def read_leader(self) -> tuple[np.ndarray, np.ndarray]:
        """리더암을 한 번 읽는다 -> (MuJoCo 목표 rad, 정규화값)."""
        start = time.perf_counter()
        action = self.teleop.get_action()
        self.read_ms = (time.perf_counter() - start) * 1e3
        target = np.array([action[f"{j}.pos"] for j in JOINT_NAMES])
        norm = np.array([self.teleop.last_normalized[j] for j in JOINT_NAMES])
        return target, norm

    def reset(self) -> None:
        """물체를 첫 키프레임으로 되돌리고 (설정돼 있으면 흩뜨리고), 팔은 지금 리더 자세에 맞춘다."""
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if self.randomizer is not None and self.randomizer.apply(self.data) is None:
            print("경고: 겹치지 않는 배치를 찾지 못해 물체를 원래 자리에 둡니다.")
        target, _ = self.read_leader()
        self.data.qpos[self.qpos_adr] = target
        self.data.qvel[self.qvel_adr] = 0.0
        self.data.ctrl[self.act_ids] = target
        mujoco.mj_forward(self.model, self.data)

    def sim_normalized(self) -> np.ndarray:
        """시뮬레이션 관절 각도를 리더와 같은 정규화 단위로."""
        return np.array([
            rad_to_normalized(j, self.data.qpos[adr], self.cfg.gripper_closed_safe)
            for j, adr in zip(JOINT_NAMES, self.qpos_adr)
        ])

    def tick(self) -> tuple[np.ndarray, np.ndarray]:
        """리더를 읽고 1/fps 초만큼 진행한다 -> (리더 정규화값, 진행 후 시뮬레이션 정규화값)."""
        target, norm = self.read_leader()
        self.data.ctrl[self.act_ids] = target
        for _ in range(self.steps_per_tick):
            mujoco.mj_step(self.model, self.data)
        return norm, self.sim_normalized()

    def close(self) -> None:
        if self.teleop.is_connected:
            self.teleop.disconnect()


class Pacer:
    """루프를 fps 에 맞춘다. 절대 시각 기준이라 sleep 이 조금씩 늦어도 평균 주기는 유지된다."""

    def __init__(self, fps: float):
        self.dt = 1.0 / fps
        self.deadline = time.perf_counter()

    def wait(self) -> None:
        self.deadline += self.dt
        rest = self.deadline - time.perf_counter()
        if rest > 0:
            time.sleep(rest)
        else:
            self.deadline = time.perf_counter()

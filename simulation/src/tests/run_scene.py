"""트롤리 씬을 MuJoCo 뷰어로 띄운다.

macOS 에서는 반드시 mjpython 으로 실행:
    .venv/bin/mjpython src/tests/run_scene.py              # 조작용 (슬라이더 있음)
    .venv/bin/mjpython src/tests/run_scene.py --plain      # 로봇용 (액추에이터 없음)
    .venv/bin/python  src/tests/run_scene.py --shot out.png --camera corner

조작은 뷰어 왼쪽 패널의 Control 섹션 슬라이더로 한다.
패널이 안 보이면 Tab 키로 연다.
    drawer_lower / drawer_upper : 서랍이 빠져나온 거리 (0 ~ 0.17 m)
    bell_press                  : 벨을 누르는 힘 (0 ~ -12 N)

키보드 단축키는 쓰지 않는다. MuJoCo 뷰어가 거의 모든 알파벳/숫자키를
렌더링 옵션 토글에 이미 할당해 두었기 때문이다.
"""

import argparse
import time
from pathlib import Path

import mujoco

# 절대경로로 연다: include 한 씬 안의 <model file="../so101/so101.xml"> 은 상대경로로 열면 못 찾는다
ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "models" / "scene"
DEMO = SCENES / "scene_trolley_demo.xml"
PLAIN = SCENES / "scene_trolley.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true",
                    help="액추에이터 없는 원본 씬 (로봇이 직접 서랍을 당기는 용도)")
    ap.add_argument("--shot", metavar="PNG", help="뷰어 대신 이미지 한 장만 렌더")
    ap.add_argument("--camera", default="overview", choices=["overview", "top", "drawers", "corner", "so101_wrist_cam"])
    args = ap.parse_args()

    scene = PLAIN if args.plain else DEMO
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print(f"loaded {scene.name}: bodies={model.nbody} joints={model.njnt} "
          f"geoms={model.ngeom} actuators={model.nu}")

    if args.shot:
        for _ in range(500):
            mujoco.mj_step(model, data)
        import imageio.v2 as iio

        # 카메라에 resolution 이 지정돼 있으면 그 크기로, 아니면 1500x1000
        w, h = model.cam_resolution[model.camera(args.camera).id]
        if w <= 1 or h <= 1:
            w, h = 1500, 1000
        with mujoco.Renderer(model, int(h), int(w)) as r:
            r.update_scene(data, camera=args.camera)
            iio.imwrite(args.shot, r.render())
        print("wrote", args.shot)
        return

    from mujoco import viewer as mj_viewer

    if model.nu:
        print("\n뷰어 왼쪽 패널 Control 섹션 슬라이더로 조작하세요 (안 보이면 Tab).")
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            lo, hi = model.actuator_ctrlrange[i]
            print(f"  {name:14s} {lo:7.3f} ~ {hi:7.3f}")

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.42, 0.24, 0.80]
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 110
        viewer.cam.elevation = -20
        while viewer.is_running():
            tic = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            rest = model.opt.timestep - (time.time() - tic)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()

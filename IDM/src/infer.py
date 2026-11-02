"""학습된 IDM 으로 액션 라벨이 없는 영상에 pseudo-label 을 붙인다.

입력 (--input):
  - mp4 폴더: 파일 하나 = 에피소드 하나 (하위 폴더까지 검색). 카메라 하나로 학습한 run 만 가능
  - LeRobot v3 데이터셋 루트 (meta/info.json 이 있으면): 학습 때와 같은 카메라를 쓴다 (--camera 로 키 이름을 바꿀 수 있음).
    action 컬럼은 없어도 된다.

라벨링은 evaluate.py 와 같은 방식(겹치지 않는 H 프레임 윈도우)이다. H 로 나누어떨어지지 않는 끝부분은
마지막 프레임에서 끝나는 윈도우 하나를 더 돌려 채우고, 다음 프레임이 없는 마지막 프레임은 직전 액션을 복사한다.

출력 (--out):
  <에피소드>.npy  float32 [프레임 수, 6]  (학습 데이터와 같은 단위)
  meta.json      관절 이름, fps, run, 에피소드별 소스 영상과 프레임 수
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from dataset import ActionNormalizer, Cache
from evaluate import JOINTS, load_model, relabel_episode
from prepare_data import cam_name, decode_video


class Clip:
    """에피소드 하나의 카메라별 프레임. relabel_episode 가 Cache 대신 받는다."""

    def __init__(self, views):
        self.views = views

    window = Cache.window


def label_frames(model, views, H, normalizer):
    n = len(views[0])
    if n <= H:
        raise ValueError(f"프레임 {n}개: 최소 {H + 1}개가 필요하다")
    clip = Clip(views)
    pred, covered = relabel_episode(model, clip, 0, n, H, normalizer)
    if covered < n - 1:
        tail, _ = relabel_episode(model, clip, n - 1 - H, n, H, normalizer)
        pred = np.concatenate([pred, tail[covered - (n - 1 - H):]])
    return np.concatenate([pred, pred[-1:]]).astype(np.float32)


def mp4_episodes(root, size):
    for vf in sorted(root.rglob("*.mp4")):
        with av.open(str(vf)) as c:
            fps = float(c.streams.video[0].average_rate)
        name = str(vf.relative_to(root).with_suffix("")).replace("/", "__")
        yield name, [decode_video(vf, size)], fps, str(vf)


def lerobot_episodes(root, keys, size):
    info = json.loads((root / "meta/info.json").read_text())
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    cache = {key: {} for key in keys}  # 카메라마다 마지막으로 디코딩한 영상 파일 하나
    for _, ep in eps.iterrows():
        views, srcs = [], []
        for key in keys:
            vf = root / info["video_path"].format(
                video_key=key,
                chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                file_index=int(ep[f"videos/{key}/file_index"]),
            )
            if vf not in cache[key]:
                cache[key].clear()
                cache[key][vf] = decode_video(vf, size)
            start = int(round(ep[f"videos/{key}/from_timestamp"] * info["fps"]))
            n = int(ep["length"])
            clip = cache[key][vf][start : start + n]
            assert len(clip) == n, f"episode {ep.episode_index} {key}: {len(clip)} != {n}"
            views.append(clip)
            srcs.append(str(vf))
        yield f"episode_{int(ep.episode_index):06d}", views, float(info["fps"]), srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train.py --save 로 저장한 runs/<이름>")
    ap.add_argument("--input", required=True, help="mp4 폴더 또는 LeRobot v3 데이터셋 루트")
    ap.add_argument("--camera", nargs="+", default=None,
                    help="LeRobot 입력의 카메라 키, 학습 때 카메라 순서대로 (기본: 학습 캐시의 키), e.g. observation.images.top")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, root, out = Path(args.run), Path(args.input), Path(args.out)
    model, cfg = load_model(run)
    H = cfg["horizon"]
    normalizer = ActionNormalizer(**cfg["normalizer"])
    train_meta = json.loads((Path(cfg["data"]) / "meta.json").read_text())
    train_cache = Cache(cfg["data"], cfg.get("cameras"))
    size, n_views = train_cache.frame_size, len(train_cache.cameras)

    if (root / "meta/info.json").exists():
        if args.camera:
            keys = args.camera
        else:
            known = train_meta.get("camera_keys") or [train_meta["camera"]]
            keys = [k for c in train_cache.cameras for k in known if cam_name(k) == c]
        if len(keys) != n_views:
            ap.error(f"카메라 {n_views}개({train_cache.cameras})로 학습한 run 인데 --camera 가 {keys}")
        args.camera = keys
        episodes = lerobot_episodes(root, keys, size)
    else:
        if n_views != 1:
            ap.error(f"mp4 폴더 입력은 카메라 하나로 학습한 run 만 가능 (이 run: {train_cache.cameras})")
        episodes = mp4_episodes(root, size)

    out.mkdir(parents=True, exist_ok=True)
    meta = {"run": str(run), "input": str(root), "camera": args.camera, "joints": JOINTS,
            "action_names": train_meta.get("action_names"), "train_fps": train_meta["fps"],
            "horizon": H, "episodes": {}}
    for name, views, fps, src in episodes:
        if abs(fps - train_meta["fps"]) > 0.5:
            print(f"경고 {name}: fps {fps:g} != 학습 fps {train_meta['fps']} (16프레임이 가리키는 시간이 달라짐)")
        try:
            action = label_frames(model, views, H, normalizer)
        except ValueError as e:
            print(f"건너뜀 {name}: {e}")
            continue
        np.save(out / f"{name}.npy", action)
        meta["episodes"][name] = {"source": src, "frames": len(action), "fps": fps}
        print(f"{name}: {len(action)} frames")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{len(meta['episodes'])}개 에피소드 -> {out}")


if __name__ == "__main__":
    main()

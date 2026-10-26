"""LeRobot v3 데이터셋을 IDM 학습용 캐시로 변환한다.

출력 (data/cache/<name>/):
  frames.npy   uint8 [N, S, S, 3]  (영상 프레임, 중앙 정사각 크롭 후 리사이즈)
  arrays.npz   action [N, 6], state [N, 6], episode_index [N], frame_index [N]
  meta.json    fps, 카메라, 에피소드 경계
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image


def decode_video(path: Path, size: int) -> np.ndarray:
    frames = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            img = fr.to_image()
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            frames.append(np.asarray(img.resize((size, size), Image.BILINEAR)))
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--camera", required=True, help="e.g. observation.images.camera_front")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]

    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    data = data.sort_values("index").reset_index(drop=True)
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)

    key = args.camera
    cache = {}
    frames_all = []
    for _, ep in eps.iterrows():
        vf = root / info["video_path"].format(
            video_key=key,
            chunk_index=int(ep[f"videos/{key}/chunk_index"]),
            file_index=int(ep[f"videos/{key}/file_index"]),
        )
        if vf not in cache:
            cache.clear()
            cache[vf] = decode_video(vf, args.size)
            print(f"decoded {vf.name}: {cache[vf].shape}")
        start = int(round(ep[f"videos/{key}/from_timestamp"] * fps))
        n = int(ep["length"])
        clip = cache[vf][start : start + n]
        assert len(clip) == n, f"episode {ep.episode_index}: {len(clip)} != {n}"
        frames_all.append(clip)

    frames = np.concatenate(frames_all)
    assert len(frames) == len(data), (len(frames), len(data))
    np.save(out / "frames.npy", frames)

    state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)[:, :6]
    action = np.stack(data["action"].to_numpy()).astype(np.float32)
    np.savez(
        out / "arrays.npz",
        action=action,
        state=state,
        episode_index=data["episode_index"].to_numpy(),
        frame_index=data["frame_index"].to_numpy(),
    )
    meta = {
        "source": str(root),
        "camera": key,
        "fps": fps,
        "num_frames": len(frames),
        "num_episodes": int(len(eps)),
        "action_names": info["features"]["action"]["names"],
        "tasks": pd.read_parquet(root / "meta/tasks.parquet").index.tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

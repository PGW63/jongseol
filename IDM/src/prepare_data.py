"""LeRobot v3 데이터셋을 IDM 학습용 캐시로 변환한다.

출력 (data/cache/<name>/):
  frames_<카메라>.npy  uint8 [N, S, S, 3]  카메라마다 하나 (영상 프레임, 중앙 정사각 크롭 후 리사이즈)
  arrays.npz          action [N, 6], state [N, 6], episode_index [N], frame_index [N]
  meta.json           fps, 카메라 목록, 에피소드 수

프레임은 에피소드 단위로 디스크(np.memmap)에 바로 쓰므로 RAM 보다 큰 데이터셋도 만들 수 있다.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image


def cam_name(key: str) -> str:
    """observation.images.top -> top"""
    return key.split(".")[-1]


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
    ap.add_argument("--camera", required=True, nargs="+",
                    help="카메라 키 하나 이상, e.g. observation.images.top observation.images.wrist")
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

    ep_of_row = data["episode_index"].to_numpy()
    for key in args.camera:
        name = cam_name(key)
        frames = np.lib.format.open_memmap(out / f"frames_{name}.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(data), args.size, args.size, 3))
        cache, off = {}, 0
        for _, ep in eps.iterrows():
            vf = root / info["video_path"].format(
                video_key=key,
                chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                file_index=int(ep[f"videos/{key}/file_index"]),
            )
            if vf not in cache:
                cache.clear()
                cache[vf] = decode_video(vf, args.size)
                print(f"[{name}] decoded {vf.name}: {cache[vf].shape}", flush=True)
            start = int(round(ep[f"videos/{key}/from_timestamp"] * fps))
            n = int(ep["length"])
            clip = cache[vf][start : start + n]
            assert len(clip) == n, f"episode {ep.episode_index}: {len(clip)} != {n}"
            assert (ep_of_row[off : off + n] == ep.episode_index).all()
            frames[off : off + n] = clip
            off += n
        assert off == len(data), (off, len(data))
        frames.flush()
        del frames

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
        "cameras": [cam_name(k) for k in args.camera],
        "camera_keys": args.camera,
        "fps": fps,
        "num_frames": len(data),
        "num_episodes": int(len(eps)),
        "action_names": info["features"]["action"]["names"],
        "tasks": pd.read_parquet(root / "meta/tasks.parquet").index.tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

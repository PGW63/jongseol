"""IDM 윈도우 데이터셋: (frame_t, frame_{t+H}) -> action[t : t+H]."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ActionNormalizer:
    """GR00T so100 data config와 같은 min-max 정규화([-1, 1])."""

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)

    @classmethod
    def fit(cls, action):
        return cls(action.min(0), action.max(0))

    def norm(self, a):
        return 2 * (a - self.lo) / np.maximum(self.hi - self.lo, 1e-6) - 1

    def denorm(self, a):
        return (np.asarray(a) + 1) / 2 * (self.hi - self.lo) + self.lo

    def to_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


class Cache:
    def __init__(self, path):
        path = Path(path)
        self.frames = np.load(path / "frames.npy", mmap_mode="r")
        arr = np.load(path / "arrays.npz")
        self.action, self.state = arr["action"], arr["state"]
        self.ep = arr["episode_index"]
        self.meta = json.loads((path / "meta.json").read_text())
        self.episodes = sorted(np.unique(self.ep).tolist())

    def episode_range(self, e):
        idx = np.nonzero(self.ep == e)[0]
        return int(idx[0]), int(idx[-1]) + 1


def split_episodes(cache, n_test, seed=0):
    """마지막이 아니라 seed 고정 셔플로 test 에피소드를 고른다."""
    eps = np.array(cache.episodes)
    rng = np.random.default_rng(seed)
    test = sorted(rng.choice(eps, n_test, replace=False).tolist())
    train = [e for e in cache.episodes if e not in test]
    return train, test


class IDMWindowDataset(Dataset):
    def __init__(self, cache, episodes, horizon, normalizer, stride=1):
        self.c, self.H, self.normalizer = cache, horizon, normalizer
        self.starts = []
        for e in episodes:
            s, t = cache.episode_range(e)
            # 마지막 프레임 t+H 가 에피소드 안에 있어야 한다
            self.starts += list(range(s, t - horizon, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        frames = np.stack([self.c.frames[s], self.c.frames[s + self.H]])  # [2, S, S, 3]
        act = self.normalizer.norm(self.c.action[s : s + self.H])
        return torch.from_numpy(frames), torch.from_numpy(act.astype(np.float32))

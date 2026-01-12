from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def record_episode(
    model,
    env,
    out_path: Path,
    max_steps: int = 1000,
    fps: int = 30,
) -> Path:
    frames = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_video(frames, out_path, fps=fps)


def record_random_episode(
    env,
    out_path: Path,
    max_steps: int = 200,
    fps: int = 30,
) -> Path:
    frames = []
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_video(frames, out_path, fps=fps)


def _write_video(frames: list, out_path: Path, fps: int = 30) -> Path:
    if not frames:
        raise ValueError("No frames captured for video.")
    out_path = Path(out_path)
    try:
        import imageio.v3 as iio
        if out_path.suffix.lower() == ".gif":
            iio.imwrite(out_path, frames, duration=1 / fps, loop=0)
            return out_path
        iio.imwrite(out_path, frames, fps=fps)
        return out_path
    except Exception:
        pass
    try:
        import imageio
        if out_path.suffix.lower() != ".gif":
            out_path = out_path.with_suffix(".gif")
        imageio.mimsave(out_path, frames, fps=fps)
        return out_path
    except Exception as exc:
        raise RuntimeError(f"Failed to write video: {exc}") from exc


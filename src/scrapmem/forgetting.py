"""Optical forgetting schedules and the JPEG transform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import shutil

from PIL import Image


@dataclass(frozen=True)
class Profile:
    quality: int
    scale: float
    stage: int


SCHEDULES = {
    "no_forget": ((100, 100, 100), (1.0, 1.0, 1.0), (180, 730)),
    "very_soft": ((95, 82, 70), (1.0, 0.95, 0.85), (180, 730)),
    "softer_old": ((90, 75, 60), (1.0, 0.90, 0.80), (180, 730)),
    "timed_gentle": ((90, 70, 40), (1.0, 0.85, 0.60), (180, 730)),
    "boundary_365": ((95, 75, 55), (1.0, 0.90, 0.75), (365, 900)),
}


def profile(policy: str, day: str, as_of: str) -> Profile:
    if policy not in SCHEDULES:
        raise ValueError(f"Unknown policy: {policy}; choose from {list(SCHEDULES)}")
    age = (date.fromisoformat(as_of) - date.fromisoformat(day)).days
    if age < 0:
        raise ValueError("as_of precedes the memory's date")
    quality, scale, boundaries = SCHEDULES[policy]
    stage = 0 if age <= boundaries[0] else 1 if age <= boundaries[1] else 2
    return Profile(quality[stage], scale[stage], stage)


def transform(
    source: Path, destination: Path, target: Profile, original_size: tuple[int, int]
):
    """Resize to the target scale, then JPEG-encode once."""
    if target.scale == 1.0 and target.quality == 100:
        shutil.copyfile(source, destination)  # No-Forget keeps the original bytes.
        return
    size = tuple(max(1, round(d * target.scale)) for d in original_size)
    with Image.open(source) as image:
        image = image.convert("RGB")
        if size[0] > image.width or size[1] > image.height:
            raise ValueError("Cannot restore resolution already removed by forgetting")
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=target.quality, subsampling=0)

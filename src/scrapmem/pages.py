"""Render a calendar day's records into a single Scrapbook Page."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .types import Record


def video_frames(path: Path, directory: Path, maximum: int = 8) -> list[Path]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Video ingestion requires ffmpeg and ffprobe on PATH")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    count = maximum
    frames = []
    for i in range(count):
        stamp = max(0.0, duration - 0.01) * i / max(1, count - 1)
        output = directory / f"frame-{i}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(stamp),
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-y",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
        if not output.is_file():
            raise RuntimeError("ffmpeg returned no video frame")
        frames.append(output)
    return frames


def wrap(draw, text: str, font, width: int) -> list[str]:
    """Character-aware wrapping handles long identifiers and non-space languages."""
    result = []
    for paragraph in text.expandtabs(4).split("\n"):
        line = ""
        for char in paragraph:
            if line and draw.textlength(line + char, font=font) > width:
                result.append(line)
                line = char
            else:
                line += char
        result.append(line)
    return result


def render_day(
    records: list[Record],
    output: Path,
    *,
    width=1400,
    font_path=None,
    max_pixels=80_000_000,
    video_max_frames=8,
) -> tuple[int, int]:
    if width < 400 or not 1 <= video_max_frames <= 8:
        raise ValueError("Page width must be >=400; video_max_frames must be 1..8")
    if not records or len({r.day for r in records}) != 1:
        raise ValueError("A page must contain records from exactly one calendar day")
    font = (
        ImageFont.truetype(str(font_path), 23)
        if font_path
        else ImageFont.load_default(size=23)
    )
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    blocks = [("text", f"SCRAPBOOK  {records[0].day}")]
    with ExitStack() as stack:
        for record in sorted(records, key=lambda r: (r.timestamp, r.id)):
            blocks.append(
                ("text", f"[{record.modality}] {record.timestamp}  |  {record.id}")
            )
            if record.text:
                blocks.append(("text", record.text))
            if record.modality in {"image", "video"}:
                media = Path(record.media_path)
                if not media.is_file():
                    raise FileNotFoundError(f"Missing media for record {record.id}")
                if record.modality == "video":
                    temporary = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    sources = video_frames(media, temporary, video_max_frames)
                else:
                    sources = [media]
                for path in sources:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        # Scale to fit inside the box; keep the full frame.
                        image.thumbnail((width - 100, 600), Image.Resampling.LANCZOS)
                        blocks.append(("image", image.copy()))
        layout = []
        height = 50
        for kind, content in blocks:
            if kind == "text":
                for line in wrap(draw, content, font, width - 100):
                    layout.append((kind, line, height))
                    height += 32
            else:
                layout.append((kind, content, height))
                height += content.height
            height += 24
        height += 30
        if width * height > max_pixels:
            raise ValueError(
                "Daily page exceeds max_pixels; raise this explicit limit or reduce media size"
            )
        canvas = Image.new("RGB", (width, height), "white")
        drawer = ImageDraw.Draw(canvas)
        for kind, content, y in layout:
            if kind == "text":
                drawer.text((50, y), content, font=font, fill="#222222")
            else:
                canvas.paste(content, (50, y))
        canvas.save(output, format="JPEG", quality=100, subsampling=0)
        return canvas.size

import io
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage


def open_image_as_srgb(image_path: str | Path | io.BytesIO) -> PilImage:
    """
    Opens an image file, applies rotation (if it's set in metadata) and converts it
    to the sRGB color space respecting the original image color space .
    Args:
        image_path: Path to the image file
    Returns:
        PIL Image in sRGB color space
    """
    exif_colorspace_srgb = 1

    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

    # Try to convert to sRGB if the image has ICC profile metadata
    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
        # Try fall back to checking EXIF
        exif_data = img.getexif()
        if exif_data is not None:
            # Assume sRGB if no ICC profile and EXIF has no ColorSpace tag
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            if color_space_value is not None and color_space_value != exif_colorspace_srgb:
                raise ValueError(
                    "Image has colorspace tag in EXIF but it isn't set to sRGB,"
                    " conversion is not supported."
                    f" EXIF ColorSpace tag value is {color_space_value}",
                )

        srgb_img = img.convert("RGB")

        # Set sRGB profile in metadata since now the image is assumed to be in sRGB.
        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img


def save_image(image_tensor: torch.Tensor, output_path: Path | str) -> None:
    """Save an image tensor to a file.
    Args:
        image_tensor: Image tensor of shape [C, H, W] or [C, 1, H, W] in range [0, 1] or [0, 255].
            C must be 3 (RGB).
        output_path: Path to save the image (any PIL-supported format, e.g., .png or .jpg)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle [C, 1, H, W] format (single frame from video tensor)
    if image_tensor.ndim == 4:
        # Squeeze frame dimension: [C, 1, H, W] -> [C, H, W]
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.squeeze(1)
        else:
            raise ValueError(f"Expected single-frame tensor with shape [C, 1, H, W], got shape {image_tensor.shape}")

    if image_tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got {image_tensor.ndim}D tensor")

    if image_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {image_tensor.shape[0]} channels")

    # Normalize to [0, 255] uint8
    if torch.is_floating_point(image_tensor) and image_tensor.max() <= 1.0:
        image_tensor = image_tensor * 255

    # Clamp to valid uint8 range to prevent overflow
    image_tensor = image_tensor.clamp(0, 255)

    # [C, H, W] -> [H, W, C]
    image_np: np.ndarray = image_tensor.permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    # Save using PIL
    Image.fromarray(image_np).save(output_path)


# ---------------------------------------------------------------------------
# Non-TTY progress helpers
# ---------------------------------------------------------------------------

def stdout_is_tty() -> bool:
    """Return True when stdout is an interactive terminal.

    Rich live-renders spinners and progress bars only in TTY mode and silently
    drops all rendering when stdout is a pipe (e.g. the ``sed | tee`` pattern
    used by preprocess_cluster.sh).  Use this to decide whether to emit
    periodic plain-text heartbeats instead.
    """
    return sys.stdout.isatty()


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


@contextmanager
def log_progress(
    description: str,
    total: int,
    log: Logger,
    *,
    interval_s: float = 10.0,
) -> Generator[Any, None, None]:
    """Context manager that yields an ``advance()`` callable.

    When stdout **is** a TTY this is a no-op (rich's own ``Progress`` widget
    handles rendering).  When stdout is **not** a TTY it emits a
    ``logger.info`` heartbeat whenever at least ``interval_s`` seconds have
    elapsed since the last emit, plus always on the first and last advance.

    Typical use::

        with log_progress("Encoding videos", total=len(dataloader), log=logger) as advance:
            for batch in dataloader:
                # ... process batch ...
                advance()

    Format of each heartbeat line::

        Encoding videos: 40/492 (8.1%) | 2.6 it/s | elapsed 0:15 | ETA 2:53
    """
    if stdout_is_tty() or total == 0:
        yield lambda: None
        return

    done = 0
    start = time.monotonic()
    last_emit = start - interval_s  # emit immediately on first advance

    def advance() -> None:
        nonlocal done, last_emit
        done += 1
        now = time.monotonic()
        elapsed = now - start
        since_last = now - last_emit
        is_first = done == 1
        is_last = done == total
        if not (is_first or is_last or since_last >= interval_s):
            return
        rate = done / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * done / total
        eta_s = (total - done) / rate if rate > 0 else 0.0
        eta_str = _fmt_duration(eta_s) if done < total else "done"
        log.info(
            "%s: %d/%d (%.1f%%) | %.2f it/s | elapsed %s | ETA %s",
            description,
            done,
            total,
            pct,
            rate,
            _fmt_duration(elapsed),
            eta_str,
        )
        last_emit = now

    yield advance


@contextmanager
def loading_heartbeat(
    description: str,
    log: Logger,
    *,
    interval_s: float = 30.0,
) -> Generator[None, None, None]:
    """Emit a periodic "still loading… Xs elapsed" log line during a long blocking load.

    This is a no-op when stdout is a TTY because rich's ``console.status``
    spinner already provides live feedback.  When stdout is a pipe (cluster
    mode) rich goes silent, leaving operators staring at a blank terminal
    for minutes while a large model loads.  This context manager spawns a
    daemon thread that fires every ``interval_s`` seconds so the log stream
    shows the process is alive.

    Typical use (wrap the ``console.status`` block)::

        if not stdout_is_tty():
            logger.info("Loading Gemma text encoder from %s ...", path)
        with loading_heartbeat("Gemma text encoder", log=logger):
            with console.status("[bold]Loading Gemma..."):
                model = load_model(path)
        if not stdout_is_tty():
            logger.info("Gemma text encoder loaded (%.1fs)", elapsed)
    """
    if stdout_is_tty():
        yield
        return

    stop_event = threading.Event()
    start = time.monotonic()

    def _heartbeat() -> None:
        tick = 0
        while not stop_event.wait(timeout=interval_s):
            tick += 1
            elapsed = time.monotonic() - start
            log.info("%s: still loading… %s elapsed", description, _fmt_duration(elapsed))

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=2)

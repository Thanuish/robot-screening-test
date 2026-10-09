"""Procedural textures shared by the MuJoCo renderer and the WebXR client.

Everything is generated with numpy at start-up (a fraction of a second), so the
repository ships no image files and both renderers show exactly the same surfaces.
"""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import mujoco
import numpy as np


def _smooth_noise(rng, h, w, cell):
    """Value noise: random grid of size (h/cell, w/cell), bilinearly upsampled."""
    gh, gw = h // cell + 2, w // cell + 2
    g = rng.random((gh, gw)).astype(np.float32)
    y = np.linspace(0, gh - 2, h, dtype=np.float32)
    x = np.linspace(0, gw - 2, w, dtype=np.float32)
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@lru_cache(maxsize=None)
def wood_floor(size: int = 1024, seed: int = 3) -> np.ndarray:
    """Oak floor planks, 4 planks across the texture (texture tile = 0.8 m)."""
    rng = np.random.default_rng(seed)
    h = w = size
    planks = 4
    pw = w // planks
    img = np.zeros((h, w, 3), np.float32)
    yy = np.arange(h, dtype=np.float32)[:, None]
    for p in range(planks):
        x0, x1 = p * pw, (p + 1) * pw
        xs = np.arange(x1 - x0, dtype=np.float32)[None, :]
        tone = rng.uniform(0.82, 1.1)
        base = np.array([0.60, 0.43, 0.28]) * tone
        # grain: stretched noise along the plank + ring-like streaks
        n1 = _smooth_noise(rng, h, x1 - x0, 64)
        n2 = _smooth_noise(rng, h // 8, x1 - x0, 3)
        n2 = np.repeat(n2, 8, axis=0)[:h]
        rings = 0.5 + 0.5 * np.sin(xs * 0.35 + 6 * n1 + 3 * n2)
        g = 0.82 + 0.12 * rings + 0.10 * (n2 - 0.5)
        plank = base[None, None, :] * g[..., None]
        # plank end joints at a random offset
        off = rng.integers(0, h)
        for k in range(2):
            jy = (off + k * h // 2) % h
            plank[max(0, jy - 1):jy + 2] *= 0.55
        img[:, x0:x1] = plank
        img[:, x0:x0 + 2] *= 0.45            # gap between planks
    img += (rng.random(img.shape).astype(np.float32) - 0.5) * 0.02
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@lru_cache(maxsize=None)
def table_laminate(size: int = 512, seed: int = 5) -> np.ndarray:
    """Light grey high-pressure laminate (lab bench) with faint mottling."""
    rng = np.random.default_rng(seed)
    n = 0.6 * _smooth_noise(rng, size, size, 48) + 0.4 * _smooth_noise(rng, size, size, 9)
    speck = rng.random((size, size)).astype(np.float32)
    base = np.array([0.74, 0.74, 0.72], np.float32)
    img = base[None, None] * (0.94 + 0.08 * n[..., None]) - 0.05 * (speck[..., None] > 0.985)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@lru_cache(maxsize=None)
def rubber_mat(size: int = 256, seed: int = 11) -> np.ndarray:
    """Dark grey anti-slip mat (not used by default; handy for new tasks)."""
    rng = np.random.default_rng(seed)
    n = _smooth_noise(rng, size, size, 6)
    img = np.array([0.16, 0.17, 0.18], np.float32)[None, None] * (0.9 + 0.2 * n[..., None])
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def decode_png(png: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(BytesIO(png)).convert("RGB"))


def encode(img: np.ndarray, fmt: str = "JPEG", quality: int = 90) -> bytes:
    from PIL import Image
    buf = BytesIO()
    im = Image.fromarray(img)
    if fmt == "JPEG":
        im.save(buf, "JPEG", quality=quality)
    else:
        im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


GENERATED_DIR = Path(__file__).resolve().parent.parent / "assets" / "generated"


def add_texture(spec: mujoco.MjSpec, name: str, img: np.ndarray) -> None:
    """Add an RGB 2D texture from a (H, W, 3) uint8 array.

    The image is cached as assets/generated/<name>.png and referenced by file, so the
    compiled model can still be saved as XML (MjSpec cannot serialise in-memory textures)."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{name}.png"
    png = encode(img, "PNG")
    if not path.exists() or path.read_bytes() != png:
        path.write_bytes(png)
    t = spec.add_texture()
    t.name = name
    t.type = mujoco.mjtTexture.mjTEXTURE_2D
    t.file = str(path)


def add_material(spec: mujoco.MjSpec, name: str, texture: str | None = None, rgba=(1, 1, 1, 1),
                 texrepeat=(1, 1), texuniform=False, specular=0.3, shininess=0.3,
                 reflectance=0.0, roughness=None, metallic=None) -> None:
    m = spec.add_material()
    m.name = name
    m.rgba = list(rgba)
    m.texrepeat = list(texrepeat)
    m.texuniform = texuniform
    m.specular = specular
    m.shininess = shininess
    m.reflectance = reflectance
    if roughness is not None:
        m.roughness = roughness
    if metallic is not None:
        m.metallic = metallic
    if texture:
        tex = list(m.textures)
        tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture
        m.textures = tex

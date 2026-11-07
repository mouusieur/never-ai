#!/usr/bin/env python3
"""
Generate official NEVER-AI ('Monolith Noir') vector & raster brand assets,
including the multi-resolution Windows executable icon (`assets/brand/never_ai.ico`)
using pure Python + NumPy (zero external image library required!).
"""

from __future__ import annotations

from pathlib import Path
import struct
import zlib
import numpy as np


BRAND_DIR = Path("assets/brand")

# Geometric Double-Blade Monogram SVG (White on Transparent / Void)
SVG_MARK_WHITE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" fill="none">
  <!-- Left Sovereign Pillar -->
  <path d="M96 84H162V428H96V84Z" fill="#E7E9EA"/>
  <!-- Right Sovereign Pillar -->
  <path d="M350 84H416V428H350V84Z" fill="#E7E9EA"/>
  <!-- Double-Rail Hollow Morphogenetic Bridge (X.com inspired hollow geometric slash) -->
  <path d="M96 84H188L416 428H324L96 84ZM144 114L344 415H372L172 114H144Z" fill="#E7E9EA"/>
</svg>
"""

SVG_APP_BADGE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" fill="none">
  <rect x="8" y="8" width="496" height="496" rx="108" fill="#000000" stroke="#2F3336" stroke-width="12"/>
  <path d="M116 108H176V404H116V108Z" fill="#FFFFFF"/>
  <path d="M336 108H396V404H336V108Z" fill="#FFFFFF"/>
  <path d="M116 108H198L396 404H314L116 108ZM158 134L332 394H356L182 134H158Z" fill="#FFFFFF"/>
</svg>
"""


def _encode_png_rgba(rgba: np.ndarray) -> bytes:
    """Encode an (H, W, 4) uint8 RGBA array into valid PNG bytes."""
    h, w, _ = rgba.shape
    raw_rows = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw_rows, level=9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _render_monogram_rgba(size: int) -> np.ndarray:
    """Anti-aliased 4x supersampled rasterization of the NEVER-AI geometric icon."""
    ss = 4
    n = size * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # Normalize coordinates to [0, 512]
    x = (xx + 0.5) * (512.0 / n)
    y = (yy + 0.5) * (512.0 / n)

    # Rounded squircle background (#000000 with #2F3336 subtle border)
    cx, cy = 256.0, 256.0
    half = 240.0
    rad = 96.0
    dx = np.maximum(np.abs(x - cx) - (half - rad), 0.0)
    dy = np.maximum(np.abs(y - cy) - (half - rad), 0.0)
    dist = np.sqrt(dx * dx + dy * dy) - rad

    inside_bg = dist <= 0.0
    border_ring = (dist <= 0.0) & (dist >= -10.0)

    # Geometric monogram coordinates
    in_y = (y >= 104.0) & (y <= 408.0)
    left_pillar = in_y & (x >= 114.0) & (x <= 172.0)
    right_pillar = in_y & (x >= 340.0) & (x <= 398.0)

    # Diagonal bridge from top-left to bottom-right with hollow core
    t = (y - 104.0) / (408.0 - 104.0)
    x_outer_left = 114.0 + t * (312.0 - 114.0)
    x_outer_right = 200.0 + t * (398.0 - 200.0)
    diag_outer = in_y & (x >= x_outer_left) & (x <= x_outer_right)

    in_hollow_y = (y >= 126.0) & (y <= 386.0)
    x_inner_left = 148.0 + t * (332.0 - 148.0)
    x_inner_right = 172.0 + t * (356.0 - 172.0)
    diag_hollow = in_hollow_y & (x >= x_inner_left) & (x <= x_inner_right)

    glyph = left_pillar | right_pillar | (diag_outer & ~diag_hollow)

    canvas = np.zeros((n, n, 4), dtype=np.float32)
    # Pitch black fill
    canvas[inside_bg, 0] = 0.0
    canvas[inside_bg, 1] = 0.0
    canvas[inside_bg, 2] = 0.0
    canvas[inside_bg, 3] = 255.0

    # Hairline border #2F3336
    canvas[border_ring, 0] = 47.0
    canvas[border_ring, 1] = 51.0
    canvas[border_ring, 2] = 54.0

    # Crisp white monogram #FFFFFF
    fg = inside_bg & glyph
    canvas[fg, 0] = 255.0
    canvas[fg, 1] = 255.0
    canvas[fg, 2] = 255.0
    canvas[fg, 3] = 255.0

    # 4x box-filter downsample for silky anti-aliased edges
    down = canvas.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    return np.clip(np.round(down), 0, 255).astype(np.uint8)


def _build_ico(sizes: list[int]) -> bytes:
    """Pack PNG-compressed frames into a standard Windows .ico container."""
    png_blobs = [_encode_png_rgba(_render_monogram_rgba(sz)) for sz in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = []
    for sz, blob in zip(sizes, png_blobs):
        dim_byte = 0 if sz >= 256 else sz
        entries.append(
            struct.pack(
                "<BBBBHHII",
                dim_byte,
                dim_byte,
                0,
                0,
                1,
                32,
                len(blob),
                offset,
            )
        )
        offset += len(blob)
    return header + b"".join(entries) + b"".join(png_blobs)


def main() -> None:
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    (BRAND_DIR / "never_mark_white.svg").write_text(SVG_MARK_WHITE, encoding="utf-8")
    (BRAND_DIR / "never_app_badge.svg").write_text(SVG_APP_BADGE, encoding="utf-8")

    png_192 = _encode_png_rgba(_render_monogram_rgba(192))
    (BRAND_DIR / "never_app_192.png").write_bytes(png_192)

    png_256 = _encode_png_rgba(_render_monogram_rgba(256))
    (BRAND_DIR / "never_app_256.png").write_bytes(png_256)

    png_512 = _encode_png_rgba(_render_monogram_rgba(512))
    (BRAND_DIR / "never_app_512.png").write_bytes(png_512)

    pwa_dir = Path("mobile_pwa")
    if pwa_dir.exists():
        (pwa_dir / "icon-192.png").write_bytes(png_192)
        (pwa_dir / "icon-512.png").write_bytes(png_512)

    ico_bytes = _build_ico([16, 32, 48, 64, 128, 256])
    (BRAND_DIR / "never_ai.ico").write_bytes(ico_bytes)
    print(f"Generated brand kit in {BRAND_DIR.resolve()} (ICO={len(ico_bytes):,} bytes)")


if __name__ == "__main__":
    main()

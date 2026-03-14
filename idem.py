#!/usr/bin/env python3
"""
idem.py — Find duplicate images and videos.

Three modes:
  Perceptual (default): computes pHash + dHash to find visually similar images.
  Video (--video): samples frames with ffmpeg to find visually similar videos,
    including re-encodes at different resolutions or bitrates. Requires ffmpeg
    on PATH. Can be combined with the default image scan.
  Exact (--exact): uses SHA-256 checksums for byte-for-byte duplicates; covers
    all media files. Reads and updates all_media_sha_hash_db.csv instead of the phash cache.

Supported image formats: JPEG, PNG, GIF, BMP, TIFF, WebP, HEIC/HEIF.
Supported video formats: MP4, MOV, AVI, MKV, WMV, WebM, FLV, etc.

Not supported in perceptual mode:
  - RAW camera files (.cr2, .nef, .arw, .dng, etc.) — use processed exports.

Usage:
    python idem.py <directory> [--threshold N] [--delta SIZE] [--cache PATH] [--review] [--page-size N] [--limit N]
    python idem.py <directory> --video [--threshold N] [--review] [--page-size N] [--limit N]
    python idem.py <directory> --exact [--review] [--page-size N] [--limit N]
    python idem.py <directory> [--exact] --interactive

Threshold guide (perceptual mode):
   0  exact visual duplicates only
   5  same image, minor JPEG re-saves
  10  same image, different resolution or format  (default)
  20  loose — risk of false positives

Threshold guide (--video mode, mean frame Hamming distance):
   0  same frames, different container or codec only
   5  same video, minor re-encode or colour grade change
  10  same video, different resolution or bitrate  (default)
  20  loose — risk of false positives
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure Unicode output works on Windows (cp1252 → utf-8).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from PIL import Image, ImageFile
    import imagehash
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # recover from broken JPEG data streams
except ImportError:
    print("Error: Pillow and imagehash are required.", file=sys.stderr)
    print("  pip install Pillow imagehash", file=sys.stderr)
    sys.exit(1)

try:
    import pybktree
except ImportError:
    print("Error: pybktree is required.", file=sys.stderr)
    print("  pip install pybktree", file=sys.stderr)
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_FILENAME    = "images_perceptual_hash_db.csv"
CACHE_FIELDS      = ["path", "size", "mtime", "phash", "dhash"]
DEFAULT_THRESHOLD = 10
_TS_TOLERANCE     = 2.0   # seconds; covers FAT32 / float round-trip noise

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".tiff", ".tif", ".webp",
    ".heic", ".heif",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm",
    ".flv", ".3gp", ".m4v", ".mts", ".m2ts",
}

MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# ── Exact-match DB constants ────────────────────────────────────────────────────

DB_FILENAME  = "all_media_sha_hash_db.csv"
DB_FIELDS    = ["path", "size", "ctime", "mtime", "checksum"]
# Sampling thresholds (must match kura.py so existing DB entries are reusable).
_SAMPLE_SIZE         = 4 * 1024 * 1024   # 4 MiB per window
_FULL_HASH_THRESHOLD = 3 * _SAMPLE_SIZE  # 12 MiB — full hash below this

# ── Video perceptual cache constants ──────────────────────────────────────────

VCACHE_FILENAME = "videos_perceptual_hash_db.csv"
VCACHE_FIELDS   = ["path", "size", "mtime", "duration", "vhash"]
N_VIDEO_FRAMES  = 8   # evenly-spaced frames sampled per video

DB_DIR = "__databases"  # subdirectory within any scanned directory that holds all 3 DB files

# ── Helpers ────────────────────────────────────────────────────────────────────

def _valid_hex(s: str) -> bool:
    """Return True if s is a non-empty valid hexadecimal string."""
    return bool(s and re.fullmatch(r'[0-9a-fA-F]+', s))


def path_without_drive(path: str) -> str:
    """Strip Windows drive letter or UNC share prefix from a path.

    C:\\Users\\foo\\bar.jpg    → \\Users\\foo\\bar.jpg
    \\\\server\\share\\foo.jpg → \\foo.jpg
    On non-Windows paths without a drive, returns the string unchanged.
    """
    p = Path(path)
    drive = p.drive
    if drive:
        return str(path)[len(drive):]
    return str(path)


def _ensure_db_dir(directory: str) -> str:
    """Return the ``__databases`` subdirectory of *directory*, creating it if needed."""
    d = os.path.join(directory, DB_DIR)
    os.makedirs(d, exist_ok=True)
    return d

# ── Cache I/O ──────────────────────────────────────────────────────────────────

def load_cache(cache_path: str) -> dict:
    """Return dict: abs_path -> {size, mtime, phash, dhash}."""
    cache = {}
    if not os.path.exists(cache_path):
        return cache
    try:
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    phash = row.get("phash", "")
                    dhash = row.get("dhash", "")
                    if not _valid_hex(phash) or (dhash and not _valid_hex(dhash)):
                        continue  # skip rows with missing or corrupt hashes
                    cache[path_without_drive(row["path"])] = {
                        "size":  int(row["size"]),
                        "mtime": float(row["mtime"]),
                        "phash": phash,
                        "dhash": dhash,
                    }
                except (ValueError, KeyError):
                    pass  # skip individual corrupt rows; rest of cache is intact
    except Exception as e:
        print(f"Warning: could not read cache ({e}), starting fresh.", file=sys.stderr)
    return cache


def save_cache(cache_path: str, cache: dict) -> None:
    """Rewrite the cache CSV from the in-memory dict (compacts any appended rows).

    Writes to a temporary file in the same directory then atomically replaces
    the target so a crash mid-write never leaves the cache truncated or empty.
    """
    try:
        cache_dir = os.path.dirname(cache_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    except Exception as e:
        print(f"Warning: could not save cache ({e}).", file=sys.stderr)
        return
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
            writer.writeheader()
            for path, data in sorted(cache.items()):
                # Keys are always drive-stripped (path_without_drive applied on insert).
                writer.writerow({"path": path, **data})
        os.replace(tmp_path, cache_path)
    except Exception as e:
        print(f"Warning: could not save cache ({e}).", file=sys.stderr)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def open_cache_for_append(cache_path: str):
    """Open the cache file for incremental appending.

    Writes the CSV header if the file does not yet exist.
    Returns an open file handle; caller is responsible for closing it.
    Each row written via this handle is immediately flushed so that an
    interrupted run loses no already-computed hashes.
    """
    f = open(cache_path, "a", newline="", encoding="utf-8")
    if f.tell() == 0:  # new or empty file — write header
        csv.DictWriter(f, fieldnames=CACHE_FIELDS).writeheader()
        f.flush()
    return f

# ── Video perceptual cache I/O ─────────────────────────────────────────────────

def load_vcache(cache_path: str) -> dict:
    """Return dict: path_without_drive -> {size, mtime, duration, vhash}.

    vhash is stored as a list of hex strings (one per sampled frame).
    """
    cache = {}
    if not os.path.exists(cache_path):
        return cache
    try:
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    vhash_str = row.get("vhash", "")
                    if not vhash_str:
                        continue
                    parts = vhash_str.split(",")
                    if not parts or not all(_valid_hex(p) for p in parts):
                        continue
                    cache[path_without_drive(row["path"])] = {
                        "size":     int(row["size"]),
                        "mtime":    float(row["mtime"]),
                        "duration": float(row["duration"]),
                        "vhash":    parts,
                    }
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"Warning: could not read video cache ({e}), starting fresh.",
              file=sys.stderr)
    return cache


def save_vcache(cache_path: str, cache: dict) -> None:
    """Atomically rewrite the video cache CSV from the in-memory dict."""
    try:
        cache_dir = os.path.dirname(cache_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    except Exception as e:
        print(f"Warning: could not save video cache ({e}).", file=sys.stderr)
        return
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VCACHE_FIELDS)
            writer.writeheader()
            for path, data in sorted(cache.items()):
                writer.writerow({
                    "path":     path_without_drive(path),
                    "size":     data["size"],
                    "mtime":    data["mtime"],
                    "duration": data["duration"],
                    "vhash":    ",".join(data["vhash"]),
                })
        os.replace(tmp_path, cache_path)
    except Exception as e:
        print(f"Warning: could not save video cache ({e}).", file=sys.stderr)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _open_vcache_for_append(cache_path: str):
    """Open the video cache file for incremental appending.

    Writes the CSV header if the file does not yet exist.
    Returns an open file handle; caller is responsible for closing it.
    """
    f = open(cache_path, "a", newline="", encoding="utf-8")
    if f.tell() == 0:
        csv.DictWriter(f, fieldnames=VCACHE_FIELDS).writeheader()
        f.flush()
    return f

# ── Exact-match DB I/O ─────────────────────────────────────────────────────────

def _load_db(db_path: str) -> dict:
    """Load all_media_sha_hash_db.csv; return dict keyed by path_without_drive."""
    entries = {}
    if not os.path.exists(db_path):
        return entries
    try:
        with open(db_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    row["size"]  = int(row["size"])
                    row["ctime"] = float(row["ctime"])
                    row["mtime"] = float(row["mtime"])
                except (TypeError, ValueError, KeyError):
                    continue
                entries[row["path"]] = row
    except Exception as e:
        print(f"Warning: could not read DB ({e}), starting fresh.", file=sys.stderr)
    return entries


def _save_db(db_path: str, entries: dict) -> None:
    """Atomically write all_media_sha_hash_db.csv from the in-memory dict."""
    try:
        db_dir = os.path.dirname(db_path) or "."
        fd, tmp = tempfile.mkstemp(dir=db_dir, suffix=".tmp")
    except Exception as e:
        print(f"Warning: could not save DB ({e}).", file=sys.stderr)
        return
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DB_FIELDS)
            writer.writeheader()
            writer.writerows(entries.values())
        os.replace(tmp, db_path)
    except Exception as e:
        print(f"Warning: could not save DB ({e}).", file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _file_checksum(path: str) -> str:
    """SHA-256 digest matching kura.py's file_checksum (sampled for large files).

    Files up to 12 MiB are hashed in full. Larger files are sampled from three
    4 MiB windows (first / middle / last) plus the file size so the algorithm
    stays compatible with checksums already stored in all_media_sha_hash_db.csv.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    # Mixing the file size into the digest first ensures two files whose sampled
    # windows happen to contain identical bytes but differ in total length will
    # produce different checksums.
    h.update(size.to_bytes(8, "little"))
    with open(path, "rb") as f:
        if size <= _FULL_HASH_THRESHOLD:
            # Use a reusable buffer + readinto to avoid allocating a new bytes
            # object on every read; cuts GC pressure on large files.
            buf = bytearray(1 << 20)
            view = memoryview(buf)
            while True:
                n = f.readinto(view)
                if not n:
                    break
                h.update(view[:n])
        else:
            h.update(f.read(_SAMPLE_SIZE))             # first 4 MiB
            f.seek(size // 2 - _SAMPLE_SIZE // 2)
            h.update(f.read(_SAMPLE_SIZE))             # middle 4 MiB
            f.seek(-_SAMPLE_SIZE, 2)
            h.update(f.read(_SAMPLE_SIZE))             # last 4 MiB
    return h.hexdigest()


# ── Hashing ────────────────────────────────────────────────────────────────────

def compute_hashes(path: str) -> tuple:
    """Return (phash_str, dhash_str). Opens the image once. Raises on error."""
    img = Image.open(path)
    img.load()  # force full JPEG decompression before hashing
    return str(imagehash.phash(img)), str(imagehash.dhash(img))

# ── Formatting / parsing ───────────────────────────────────────────────────────

def parse_size(s: str) -> int:
    """Parse a human-readable size string into bytes.

    Accepts an optional suffix (case-insensitive): b, kb, mb, gb, tb.
    No suffix means bytes. Examples: '500', '50kb', '2.5MB', '1GB'.
    """
    s = s.strip()
    lo = s.lower()
    multipliers = [("tb", 1024**4), ("gb", 1024**3), ("mb", 1024**2), ("kb", 1024), ("b", 1)]
    for suffix, mult in multipliers:
        if lo.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)  # bare number → bytes


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"

# ── Terminal helpers ────────────────────────────────────────────────────────────

def _fmt_path(path: str) -> str:
    """Return a file:// URI on Windows (clickable in terminal), plain path elsewhere."""
    if sys.platform == "win32":
        return Path(path).as_uri()
    return path


def _terminal_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _progress_bar(i, n, label=""):
    """Overwrite the current line with a progress bar. No-op if not a TTY."""
    if not sys.stdout.isatty() or n == 0:
        return
    cols = _terminal_width()
    counter  = f"{i}/{n}"
    # Fixed overhead: "  [" bar "]  " counter "  " = bar_area + len(counter) + 9
    bar_area = min(40, max(10, cols - len(counter) - 9))
    filled   = round(i / n * bar_area)
    bar      = "#" * filled + "-" * (bar_area - filled)
    head     = f"  [{bar}]  {counter}"
    avail    = cols - len(head) - 3
    tail     = ("  " + label[:avail]) if avail > 4 and label else ""
    sys.stdout.write(("\r" + head + tail).ljust(cols - 1)[:cols - 1])
    sys.stdout.flush()


def _clear_bar():
    """Blank the current progress-bar line so an error message can be printed."""
    if not sys.stdout.isatty():
        return
    cols = _terminal_width()
    sys.stdout.write("\r" + " " * (cols - 1) + "\r")
    sys.stdout.flush()


# ── Core logic ─────────────────────────────────────────────────────────────────

def scan_files(directory: str, extensions=IMAGE_EXTENSIONS) -> list:
    """Return sorted list of absolute paths for matching files under directory.

    ``__duplicate_files_trash/`` and ``__databases/`` are excluded so previously
    trashed files are never re-presented as duplicates, and DB temp files are
    never mistaken for media files.
    """
    result = []
    _skip = {"__duplicate_files_trash", DB_DIR}
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _skip]  # in-place edit prunes os.walk's descent
        for fname in files:
            if Path(fname).suffix.lower() in extensions:
                result.append(os.path.join(root, fname))
    return sorted(result)


def scan_exact_files(directory: str) -> list:
    """Return sorted list of all media files (images + videos) under directory.

    Convenience wrapper around scan_files for exact/checksum mode.
    """
    return scan_files(directory, MEDIA_EXTENSIONS)


def build_hashes(all_files: list, cache: dict, cache_out=None) -> tuple:
    """
    Compute/load perceptual hashes (phash + dhash) for all files.
    Returns (hashes dict, new_count, rehashed_count, error_count).
    hashes maps path -> (phash_int, dhash_int, size_bytes).
    Mutates cache in-place with any newly computed hashes.

    cache_out: optional open file handle (from open_cache_for_append). When
    provided, newly computed hashes are appended and flushed to disk every
    200 updates (and once more at the end) so that an interrupted run loses
    at most 200 hashes worth of work.
    """
    writer = csv.DictWriter(cache_out, fieldnames=CACHE_FIELDS) if cache_out else None
    hashes = {}
    new_count = rehashed_count = errors = 0
    n = len(all_files)

    for i, path in enumerate(all_files, 1):
        if i % 50 == 0 or i == n:
            _progress_bar(i, n, os.path.basename(path))

        try:
            st = os.stat(path)
        except OSError as e:
            _clear_bar()
            print(f"  Warning: skipped {os.path.basename(path)}: {e}", flush=True)
            errors += 1
            continue

        size, mtime = st.st_size, st.st_mtime
        cache_key = path_without_drive(path)
        cached = cache.get(cache_key)

        metadata_ok = (cached and cached["size"] == size
                       and abs(cached["mtime"] - mtime) < _TS_TOLERANCE)

        is_new = not cached

        if metadata_ok:
            # Cache hit — file unchanged (size and mtime within tolerance).
            ph = int(cached["phash"], 16)
            dh = int(cached["dhash"], 16)
        else:
            try:
                phash_str, dhash_str = compute_hashes(path)
            except Exception as e:
                _clear_bar()
                print(f"  Warning: skipped {os.path.basename(path)}: {e}", flush=True)
                errors += 1
                continue
            ph = int(phash_str, 16)
            dh = int(dhash_str, 16)
            cache[cache_key] = {"size": size, "mtime": mtime,
                                "phash": phash_str, "dhash": dhash_str}
            if writer:
                writer.writerow({"path": cache_key, "size": size, "mtime": mtime,
                                 "phash": phash_str, "dhash": dhash_str})
            if is_new:
                new_count += 1
            else:
                rehashed_count += 1
            if writer and (new_count + rehashed_count) % 200 == 0:
                cache_out.flush()  # periodic flush: limits lost work to ~200 hashes on interrupt

        hashes[path] = (ph, dh, size)

    if writer:
        cache_out.flush()  # flush any remainder at the end
    if n > 0:
        print()  # end progress line
    return hashes, new_count, rehashed_count, errors


def group_duplicates(hashes: dict, threshold: int) -> list:
    """
    Return list of duplicate groups, each a list of (path, size_bytes).
    Uses a BK-tree for O(n log n) near-duplicate detection rather than O(n²).
    Within each group files are sorted largest-first (largest = likely original).
    Groups are sorted most-files-first.
    """
    if not hashes:
        return []

    def hamming(a, b):
        # Items are (path, ph_int) tuples; XOR the integer pHashes and count
        # the differing bits to get the Hamming distance between the two images.
        return (a[1] ^ b[1]).bit_count()

    items = [(path, ph) for path, (ph, _, _sz) in hashes.items()]
    tree  = pybktree.BKTree(hamming, items)

    seen   = set()
    groups = []
    n = len(items)

    for i, (path, (ph, dh, _sz)) in enumerate(hashes.items(), 1):
        if i % 50 == 0 or i == n:
            _progress_bar(i, n)
        if path in seen:
            continue
        matches = tree.find((path, ph), threshold)  # includes self
        if len(matches) < 2:
            continue
        # Secondary dhash filter: pHash captures frequency-domain (DCT) structure;
        # dHash captures pixel-gradient direction changes.  Requiring both hashes
        # to be within threshold significantly reduces false positives, since a
        # pair of unrelated images is unlikely to fool both independently.
        # Candidates already assigned to an earlier group are excluded so that each
        # image appears in exactly one group (the first pivot whose BK-tree search
        # returned it).
        group_paths = [m[1][0] for m in matches
                       if m[1][0] not in seen
                       and (hashes[m[1][0]][1] ^ dh).bit_count() <= threshold]
        if len(group_paths) < 2:
            continue
        seen.update(group_paths)
        group = sorted(
            ((p, hashes[p][2]) for p in group_paths),
            key=lambda x: x[1],
            reverse=True,
        )
        groups.append(group)

    print()  # end progress line
    groups.sort(key=lambda g: (-len(g), -g[0][1]))
    return groups

# ── Video perceptual hashing ───────────────────────────────────────────────────

def ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def _get_video_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe, falling back to ffmpeg."""
    import subprocess

    if shutil.which("ffprobe"):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            try:
                return float(r.stdout.strip())
            except ValueError:
                pass

    # Fallback: parse "Duration: HH:MM:SS.ss" from ffmpeg -i stderr.
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path],
        capture_output=True, text=True, timeout=30,
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s
    raise RuntimeError(f"Could not determine duration of {os.path.basename(path)}")


def compute_video_hashes(path: str, n: int = N_VIDEO_FRAMES) -> tuple:
    """Return (duration_seconds, [phash_int, ...]) for a video file.

    Opens the video n times in one ffmpeg call, each with a fast input-level
    seek (-ss before -i) to a midpoint timestamp.  trim=end_frame=1 stops
    ffmpeg from reading past the first frame of each seek, so only ~1 GOP
    worth of data is decoded per sample regardless of video length.
    Output is raw 64×64 RGB — no PNG encode/decode overhead.
    Raises RuntimeError if ffmpeg fails or the video cannot be decoded.
    """
    import subprocess

    duration = _get_video_duration(path)
    if duration <= 0:
        raise RuntimeError(
            f"Invalid duration ({duration:.1f}s) for {os.path.basename(path)}"
        )

    FRAME_W, FRAME_H = 64, 64
    timestamps = [duration * (2 * i + 1) / (2 * n) for i in range(n)]

    # Placing -ss *before* -i tells ffmpeg to seek at the input level: it jumps
    # to the nearest preceding keyframe (fast).  Putting -ss *after* -i would
    # force ffmpeg to decode every frame from the file start up to the target
    # timestamp, which is prohibitively slow for large or high-bitrate videos.
    args = ["ffmpeg", "-nostdin"]
    for t in timestamps:
        args += ["-ss", f"{t:.6f}", "-i", path]

    # One frame per input, scaled; then concatenate into a single stream.
    fc_parts = [
        f"[{i}:v]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS,"
        f"scale={FRAME_W}:{FRAME_H}[v{i}]"
        for i in range(n)
    ]
    fc_concat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1[out]"
    args += [
        "-filter_complex", ";".join(fc_parts) + ";" + fc_concat,
        "-map", "[out]",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1", "-loglevel", "error",
    ]

    result = subprocess.run(args, capture_output=True, timeout=120)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            f"ffmpeg failed for {os.path.basename(path)}"
        )

    frame_size = FRAME_W * FRAME_H * 3  # bytes per raw RGB frame: 64×64×3 = 12 288
    raw = result.stdout
    n_got = len(raw) // frame_size
    if n_got == 0:
        raise RuntimeError(
            f"ffmpeg produced no frames for {os.path.basename(path)}"
        )

    frame_hashes = []
    for i in range(min(n, n_got)):
        chunk = raw[i * frame_size : (i + 1) * frame_size]
        img = Image.frombytes("RGB", (FRAME_W, FRAME_H), chunk)
        frame_hashes.append(int(str(imagehash.phash(img)), 16))

    return duration, frame_hashes


def build_video_hashes(files: list, cache: dict, cache_out=None) -> tuple:
    """Compute/load frame hashes for video files.

    Returns (vhashes dict, new_count, rehashed_count, error_count).
    vhashes maps path -> (duration, [phash_int, ...], size_bytes).
    Mutates cache in-place with any newly computed hashes.
    """
    writer = csv.DictWriter(cache_out, fieldnames=VCACHE_FIELDS) if cache_out else None
    vhashes = {}
    new_count = rehashed_count = errors = 0
    n = len(files)

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        _progress_bar(i, n, name)

        try:
            st = os.stat(path)
        except OSError as e:
            _clear_bar()
            print(f"  Warning: skipped {name}: {e}", flush=True)
            errors += 1
            continue

        size, mtime = st.st_size, st.st_mtime
        cache_key = path_without_drive(path)
        cached = cache.get(cache_key)
        metadata_ok = (cached and cached["size"] == size
                       and abs(cached["mtime"] - mtime) < _TS_TOLERANCE)
        is_new = not cached

        if metadata_ok:
            dur = cached["duration"]
            frames = [int(h, 16) for h in cached["vhash"]]
        else:
            try:
                dur, frames = compute_video_hashes(path)
            except Exception as e:
                _clear_bar()
                print(f"  Warning: skipped ({fmt_size(size).strip()}) {_fmt_path(path)}", flush=True)
                errors += 1
                continue
            hex_frames = [format(h, "016x") for h in frames]
            cache[cache_key] = {
                "size": size, "mtime": mtime,
                "duration": dur, "vhash": hex_frames,
            }
            if writer:
                writer.writerow({
                    "path": cache_key, "size": size, "mtime": mtime,
                    "duration": dur, "vhash": ",".join(hex_frames),
                })
                # Flush after every video: each hash takes several seconds, so a
                # per-entry flush (rather than the 200-entry batch used for images)
                # keeps data-loss on interrupt to at most one video's worth of work.
                cache_out.flush()
            if is_new:
                new_count += 1
            else:
                rehashed_count += 1

        vhashes[path] = (dur, frames, size)

    print()  # end progress line
    return vhashes, new_count, rehashed_count, errors


def _video_distance(frames_a: list, frames_b: list) -> float:
    """Mean per-frame pHash Hamming distance between two frame sequences."""
    n = min(len(frames_a), len(frames_b))
    if n == 0:
        return float("inf")
    return sum((frames_a[i] ^ frames_b[i]).bit_count() for i in range(n)) / n


def group_video_duplicates(vhashes: dict, threshold: int) -> list:
    """Group videos by perceptual similarity using mean frame Hamming distance.

    Pre-filters candidate pairs by duration: videos must be within
    max(10 s, 5% of the longer video) of each other. This targets re-encodes
    and resolution variants; trimmed or offset clips will not match.

    Returns groups in the same format as group_duplicates:
    list of [(path, size_bytes), ...] largest-first, most-files-first.
    """
    if not vhashes:
        return []

    # Sort by duration for the early-break inner loop.
    paths = sorted(vhashes.keys(), key=lambda p: vhashes[p][0])

    # Union-Find with path compression: groups transitively similar videos so that
    # if A~B and B~C they end up in the same component even when A and C are never
    # directly compared.  A simple "add to list" approach would miss such chains.
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    n = len(paths)
    for i in range(n):
        pa = paths[i]
        dur_a, frames_a, _ = vhashes[pa]
        for j in range(i + 1, n):
            pb = paths[j]
            dur_b, frames_b, _ = vhashes[pb]
            # dur_b >= dur_a (sorted ascending).
            # tol grows with dur_b, so once the gap exceeds it the break holds
            # for all larger dur_b values (monotone).
            tol = max(10.0, 0.05 * dur_b)
            if dur_b - dur_a > tol:
                break
            if _video_distance(frames_a, frames_b) <= threshold:
                union(pa, pb)

    groups_map: dict = {}
    for p in paths:
        groups_map.setdefault(find(p), []).append(p)

    groups = []
    for members in groups_map.values():
        if len(members) < 2:
            continue
        group = sorted(
            [(p, vhashes[p][2]) for p in members],
            key=lambda x: x[1],
            reverse=True,
        )
        groups.append(group)

    groups.sort(key=lambda g: (-len(g), -g[0][1]))
    return groups

# ── Exact-match logic ──────────────────────────────────────────────────────────

def _collect_stale(files: list, db: dict) -> list:
    """Return paths from *files* needing checksum computation (new or stale).

    Stale entries are removed from *db* in-place only when the file is confirmed
    accessible and its metadata has changed.  Transient OSErrors are ignored so
    the cached entry survives until the next successful stat.
    """
    to_compute = []
    for path in files:
        k = path_without_drive(path)
        cached = db.get(k)
        if cached:
            try:
                st = os.stat(path)
                if (cached["size"] == st.st_size
                        and abs(cached["mtime"] - st.st_mtime) < _TS_TOLERANCE):
                    continue  # cache hit
                del db[k]  # stale — metadata changed
            except OSError:
                pass
        to_compute.append(path)
    return to_compute


def build_exact_index(all_files: list, db_path: str) -> tuple:
    """Load/update all_media_sha_hash_db.csv; compute checksums for new or stale files.

    Returns (db, new_count, gone_count, errors).
    db is keyed by path_without_drive.
    """
    db = _load_db(db_path)
    live_keys = {path_without_drive(p) for p in all_files}

    # Remove entries for files no longer on disk.
    gone_count = 0
    gone = [k for k in db if k not in live_keys]
    for k in gone:
        del db[k]
        gone_count += 1

    # Identify new files and stale entries (metadata changed).
    to_compute = _collect_stale(all_files, db)

    new_count = errors = 0
    n = len(to_compute)
    for i, path in enumerate(to_compute, 1):
        if i % 50 == 0 or i == n:
            _progress_bar(i, n, os.path.basename(path))
        try:
            st = os.stat(path)
            checksum = _file_checksum(path)
        except Exception as e:
            _clear_bar()
            print(f"  Warning: skipped {os.path.basename(path)}: {e}", flush=True)
            errors += 1
            continue
        k = path_without_drive(path)
        db[k] = {
            "path": k,
            "size": st.st_size,
            "ctime": st.st_ctime,
            "mtime": st.st_mtime,
            "checksum": checksum,
        }
        new_count += 1

    if n > 0:
        print()  # end progress line
    _save_db(db_path, db)
    return db, new_count, gone_count, errors


def group_exact_duplicates(db: dict, abs_path_map: dict) -> list:
    """Group files in db by checksum into duplicate groups.

    db: {path_without_drive -> entry}
    abs_path_map: {path_without_drive -> abs_path}
    Returns list of groups, each group = [(abs_path, size), ...] largest-first.
    """
    by_checksum: dict = {}
    for k, entry in db.items():
        by_checksum.setdefault(entry["checksum"], []).append((k, entry["size"]))

    groups = []
    for items in by_checksum.values():
        if len(items) < 2:
            continue
        group = sorted(
            [(abs_path_map.get(k, k), size) for k, size in items],
            key=lambda x: x[1],
            reverse=True,
        )
        groups.append(group)

    groups.sort(key=lambda g: (-len(g), -g[0][1]))
    return groups


def _update_checksums_additive(files: list, db_path: str) -> tuple:
    """Load db_path, compute checksums for new/stale files in *files*, save.

    Unlike build_exact_index, entries for files *not* in *files* are preserved
    (so image entries in a shared all_media_sha_hash_db.csv are not removed when only videos
    are being indexed).  Returns (db, new_count, errors).
    """
    db = _load_db(db_path)
    to_compute = _collect_stale(files, db)

    new_count = errors = 0
    n = len(to_compute)
    for i, path in enumerate(to_compute, 1):
        if i % 50 == 0 or i == n:
            _progress_bar(i, n, os.path.basename(path))
        try:
            st = os.stat(path)
            checksum = _file_checksum(path)
        except Exception as e:
            _clear_bar()
            print(f"  Warning: skipped {os.path.basename(path)}: {e}", flush=True)
            errors += 1
            continue
        k = path_without_drive(path)
        db[k] = {"path": k, "size": st.st_size, "ctime": st.st_ctime,
                  "mtime": st.st_mtime, "checksum": checksum}
        new_count += 1

    if n > 0:
        print()
    _save_db(db_path, db)
    return db, new_count, errors


# ── Output ─────────────────────────────────────────────────────────────────────

def _exact_group_keep(group: list, ignore: tuple, directory: str) -> tuple:
    """Return (keep_path, final_path) for an exact-mode duplicate group.

    keep_path  — the replica with the best-scored folder (the file left in place).
    final_path — keep_path's directory combined with the best filename from any
                 replica.  When a file in a well-named folder has a generic name,
                 it can adopt a more descriptive name from another replica without
                 moving to a different directory.
    """
    files = [{"path": p, "name": os.path.basename(p), "dir": str(Path(p).parent)}
             for p, _ in group]
    keep_path = _pick_keeper(files, ignore, directory)
    best_name = os.path.basename(keep_path)
    best_ns = _name_score(best_name, ignore)
    for f in files:
        ns = _name_score(f["name"], ignore)
        if ns > best_ns:
            best_ns, best_name = ns, f["name"]
    return keep_path, os.path.join(str(Path(keep_path).parent), best_name)


def _print_group_files(group: list, keep_path: str, final_path: str) -> None:
    """Print keep/remove lines for an exact-mode group."""
    for path, _ in sorted(group, key=lambda x: x[0] == keep_path):
        label = "keep  " if path == keep_path else "remove"
        print(f"  {label}  {path}")
        if path == keep_path and os.path.normpath(final_path) != os.path.normpath(keep_path):
            print(f"            -> {final_path}")


def print_results(groups: list, directory: str,
                  exact: bool = False, ignore: tuple = ()) -> None:
    if not groups:
        print("\nNo duplicate groups found.")
        return

    total_files   = sum(len(g) for g in groups)
    # Bytes that could be freed if all but the largest copy were removed
    wasted_bytes  = sum(size for g in groups for _, size in g[1:])

    print(f"\nFound {len(groups)} duplicate group(s)  *  "
          f"{total_files} files  *  "
          f"{fmt_size(wasted_bytes).strip()} potentially recoverable\n")
    print("-" * 80)

    same_folder_groups = []  # exact-mode groups where all replicas share one folder
    displayed = 0

    for group in groups:
        if exact:
            dirs = {str(Path(p).parent) for p, _ in group}
            if len(dirs) == 1:
                same_folder_groups.append(group)
                continue
        displayed += 1
        print(f"\nGroup {displayed}  *  {len(group)} files")
        if exact:
            keep_path, final_path = _exact_group_keep(group, ignore, directory)
            _print_group_files(group, keep_path, final_path)
        else:
            for j, (path, size) in enumerate(group):
                tag = "  <- largest" if j == 0 else ""
                rel = os.path.relpath(path, directory)
                print(f"  {fmt_size(size)}  {rel}{tag}")
                # Full path on a second line for easy copy-paste
                print(f"           {path}")

    if same_folder_groups:
        n = len(same_folder_groups)
        print(f"\n{n} group(s) not shown (same folder — use --review or --interactive to resolve):")
        for group in same_folder_groups:
            keep_path, _ = _exact_group_keep(group, ignore, directory)
            print(f"  (same folder: {str(Path(keep_path).parent)})  *  {len(group)} files"
                  f"  —  keep {os.path.basename(keep_path)}")

    print()

# ── Review UI ──────────────────────────────────────────────────────────────────

def _trash_path(abs_path: str, directory: str) -> Path:
    """Return destination path inside __duplicate_files_trash/, preserving relative structure."""
    return Path(directory) / "__duplicate_files_trash" / Path(abs_path).relative_to(directory)


def _unique_dst(dst: Path) -> Path:
    """Return dst, or dst with a _N counter suffix if it already exists."""
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    counter = 1
    while dst.exists():
        dst = dst.parent / f"{stem}_{counter}{suffix}"
        counter += 1
    return dst


# Noise tokens stripped when scoring image filenames.
_NAME_NOISE = {
    'PXL', 'IMG', 'DSC', 'DSCF', 'DSCN', 'DCIM', 'WA', 'VID', 'MVI',
    'FB', 'PICT', 'IMGP', 'IMGD', 'DCF',
    'WHATSAPP', 'IMAGES', 'PHOTO', 'PHOTOS', 'IMAGE', 'BACKUP', 'BKP',
}

# Noise tokens stripped when scoring folder names.
_FOLDER_NOISE = {
    'PHOTO', 'PHOTOS', 'IMAGE', 'IMAGES', 'WHATSAPP', 'BACKUP', 'BKP',
}


def _token_score(text: str, noise: set, extra_noise=()) -> int:
    """Count meaningful alphabetic characters after stripping noise tokens.

    If the entire *text* exactly matches an extra_noise pattern (case-insensitive),
    scores zero immediately. Otherwise, individual tokens that exactly match an
    extra_noise pattern are stripped before counting.
    """
    if any(p.lower() == text.lower() for p in extra_noise):
        return 0
    tokens = [t for t in re.split(r'[\s_\-\d]+', text) if t]
    meaningful = [t for t in tokens if t.upper() not in noise]
    meaningful = [t for t in meaningful if not any(t.lower() == p.lower() for p in extra_noise)]
    return len(re.sub(r'[^a-zA-Z]', '', ''.join(meaningful)))


def _name_score(filename: str, extra_noise=()) -> int:
    """Score how 'meaningful' an image filename is (higher = more English words)."""
    return _token_score(Path(filename).stem, _NAME_NOISE, extra_noise)


def _folder_score(dir_path: str, extra_noise=(), base: str = '') -> int:
    """Sum scores of all folder components between base and dir_path.

    If base is given, scores every component of the relative sub-path; otherwise
    only the immediate folder name is scored (legacy behaviour).
    Extra-noise tokens are stripped from each component before scoring (via
    _token_score); a component whose entire name matches a noise word scores 0.
    """
    if not dir_path:
        return 0
    if base:
        try:
            parts = Path(dir_path).relative_to(base).parts
        except ValueError:
            parts = (Path(dir_path).name,)
    else:
        parts = (Path(dir_path).name,)
    for part in parts:
        if any(p.lower() == part.lower() for p in extra_noise):
            return -1
    return sum(_token_score(part, _FOLDER_NOISE, extra_noise) for part in parts)


def _smart_defaults(files: list, extra_noise=(), base: str = '') -> tuple:
    """Return smart UI defaults for a duplicate group (sorted largest-first).

    Returns (keep_path, rename_src_or_None, folder_src_or_None).
    Mirrors the init() auto-selection logic in the review UI.
    Each element of *files* must be a dict with keys 'path', 'name', 'dir'.
    """
    kept = files[0]
    best_n_score, best_n_path = _name_score(kept['name'], extra_noise), None
    best_f_score, best_f_path = _folder_score(kept['dir'], extra_noise, base), None
    for f in files[1:]:
        ns = _name_score(f['name'], extra_noise)
        if ns > best_n_score:
            best_n_score, best_n_path = ns, f['path']
        fs = _folder_score(f['dir'], extra_noise, base)
        if fs > best_f_score:
            best_f_score, best_f_path = fs, f['path']
    return kept['path'], best_n_path, best_f_path


def _pick_keeper(files: list, ignore: tuple = (), base: str = '') -> str:
    """Return the path of the best file to keep from a duplicate group.

    Ranks by (folder_score, name_score); ties go to the first (largest) file.
    Each element of *files* must be a dict with keys 'path', 'name', 'dir'.
    """
    best = files[0]
    best_score = (_folder_score(best['dir'], ignore, base), _name_score(best['name'], ignore))
    for f in files[1:]:
        s = (_folder_score(f['dir'], ignore, base), _name_score(f['name'], ignore))
        if s > best_score:
            best_score, best = s, f
    return best['path']


def _safe_relpath(path: str, base: str) -> str:
    """Return relpath(path, base), or path itself if they are on different drives."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def _groups_to_json(groups: list, directory: str,
                    ignore: tuple = (), base: str = '') -> list:
    """Convert internal groups list to a JSON-serialisable structure.

    Embeds server-computed smart defaults (keep_default, rename_src, folder_src)
    so the browser UI can read them directly without re-implementing scoring in JS.
    """
    result = []
    for group in groups:
        files = []
        for path, size in group:
            parent = str(Path(path).parent)
            files.append({
                "path": path,
                "rel": _safe_relpath(path, directory),
                "name": os.path.basename(path),
                "dir": parent,
                "dir_rel": _safe_relpath(parent, directory),
                "size": size,
                "size_fmt": fmt_size(size).strip(),
                "is_video": Path(path).suffix.lower() in VIDEO_EXTENSIONS,
            })
        keep_default, rename_src, folder_src = _smart_defaults(files, ignore, base)
        result.append({
            "files": files,
            "keep_default": keep_default,
            "rename_src": rename_src,
            "folder_src": folder_src,
        })
    return result


_REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>idem — Review Duplicates</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #f0f0f0; color: #222; }
header {
  background: #1a1a2e; color: #eee; padding: 12px 20px;
  display: flex; align-items: center; gap: 16px;
  position: sticky; top: 0; z-index: 10; box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
header h1 { font-size: 1.05rem; white-space: nowrap; }
#stats { font-size: 0.88rem; opacity: 0.75; flex: 1; }
#confirm-btn {
  background: #e94560; color: #fff; border: none;
  padding: 8px 18px; border-radius: 6px; cursor: pointer; font-size: 0.9rem;
  white-space: nowrap;
}
#confirm-btn:hover { background: #c73652; }
main { max-width: 1200px; margin: 0 auto; padding: 20px; }
.group-card {
  background: #fff; border-radius: 8px; padding: 16px;
  margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
.group-header { font-weight: 600; margin-bottom: 12px; color: #555; font-size: 0.88rem; }
.file-slots { display: flex; flex-wrap: wrap; gap: 14px; }
.file-slot {
  display: flex; flex-direction: column; align-items: flex-start;
  width: 220px; position: relative; transition: opacity 0.15s;
}
.file-slot img {
  width: 220px; height: 195px; object-fit: cover; border-radius: 6px;
  cursor: pointer; border: 2px solid transparent; display: block;
}
.file-slot img:hover { border-color: #e94560; }
.file-slot.deselected img { opacity: 0.3; filter: grayscale(70%); }
.file-slot.deselected .slot-info { opacity: 0.45; }
.original-badge {
  position: absolute; top: 4px; left: 4px;
  background: #2e7d32; color: #fff;
  font-size: 0.62rem; padding: 2px 6px; border-radius: 4px;
  pointer-events: none; letter-spacing: 0.02em;
}
.img-placeholder {
  width: 220px; height: 195px; background: #e0e0e0; border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; color: #777; text-align: center; padding: 10px;
  word-break: break-all;
}
.slot-info { width: 220px; margin-top: 6px; }
.slot-name { font-size: 0.76rem; font-weight: 600; word-break: break-all; line-height: 1.3; }
.slot-path { font-size: 0.67rem; color: #999; word-break: break-all; margin-top: 2px; line-height: 1.3; }
.slot-size { font-size: 0.72rem; color: #666; margin-top: 3px; }
.slot-keep {
  display: flex; align-items: center; gap: 6px;
  margin-top: 6px; font-size: 0.8rem; cursor: pointer;
}
.slot-keep input { cursor: pointer; }
.pagination {
  display: flex; align-items: center; gap: 14px;
  justify-content: center; margin-top: 24px; margin-bottom: 32px;
}
.pagination button {
  padding: 6px 16px; border: 1px solid #ccc; background: #fff;
  border-radius: 4px; cursor: pointer; font-size: 0.9rem;
}
.pagination button:disabled { opacity: 0.35; cursor: default; }
#page-info { font-size: 0.88rem; color: #666; }
#overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.55); z-index: 100;
  align-items: center; justify-content: center;
}
#overlay.active { display: flex; }
#dialog {
  background: #fff; border-radius: 12px; padding: 36px 44px;
  max-width: 460px; width: 90%; text-align: center;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
#dialog h2 { margin-bottom: 18px; font-size: 1.3rem; }
#dialog p { margin-bottom: 6px; font-size: 0.95rem; color: #444; }
#done-btn {
  margin-top: 22px; background: #1a1a2e; color: #fff; border: none;
  padding: 10px 26px; border-radius: 6px; cursor: pointer; font-size: 1rem;
}
#done-btn:hover { background: #2a2a4e; }
#done-btn:disabled { opacity: 0.5; cursor: default; }
#hover-preview {
  display: none; position: fixed; z-index: 200; pointer-events: none;
  border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.55);
  background: #111; overflow: hidden;
}
#hover-preview img { display: block; max-width: 1024px; max-height: 1024px; object-fit: contain; }
.slot-use-name-btn {
  margin-top: 5px; font-size: 0.75rem; color: #1976d2;
  background: none; border: 1px solid #90caf9; border-radius: 4px;
  padding: 2px 8px; cursor: pointer; width: 100%;
  text-align: left; transition: background 0.15s;
}
.slot-use-name-btn:hover:not(:disabled) { background: #e3f2fd; }
.slot-use-name-btn.active { background: #1976d2; color: #fff; border-color: #1976d2; }
.slot-use-name-btn:disabled { opacity: 0.3; cursor: default; }
.slot-rename-hint { font-size: 0.7rem; color: #1976d2; margin-top: 4px; font-style: italic; }
.slot-use-folder-btn {
  margin-top: 3px; font-size: 0.75rem; color: #388e3c;
  background: none; border: 1px solid #a5d6a7; border-radius: 4px;
  padding: 2px 8px; cursor: pointer; width: 100%;
  text-align: left; transition: background 0.15s;
}
.slot-use-folder-btn:hover:not(:disabled) { background: #e8f5e9; }
.slot-use-folder-btn.active { background: #388e3c; color: #fff; border-color: #388e3c; }
.slot-use-folder-btn:disabled { opacity: 0.3; cursor: default; }
.slot-folder-hint { font-size: 0.7rem; color: #388e3c; margin-top: 2px; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>idem &mdash; Duplicate Review</h1>
  <span id="stats">Loading&hellip;</span>
  <button id="confirm-btn" onclick="confirmSelections()">Confirm &amp; Move to Trash</button>
</header>
<main>
  <div id="groups-container"></div>
  <div class="pagination">
    <button id="prev-btn" onclick="changePage(-1)">&#8592; Prev</button>
    <span id="page-info"></span>
    <button id="next-btn" onclick="changePage(1)">Next &#8594;</button>
  </div>
</main>
<div id="hover-preview"><img id="hover-preview-img" alt=""></div>
<div id="overlay">
  <div id="dialog">
    <h2 id="dialog-title">Done</h2>
    <p id="dialog-moved"></p>
    <p id="dialog-renamed"></p>
    <p id="dialog-skipped"></p>
    <p id="dialog-errors"></p>
    <button id="done-btn" onclick="doneProceed()">Done</button>
  </div>
</div>
<script>
let groups = [];
let currentPage = 0;
const PAGE_SIZE = /*PAGESIZE*/10/*PAGESIZE*/;
const selections = new Map(); // groupIdx -> Set of paths to KEEP
const renames = new Map();     // groupIdx -> path of file whose name to use (null = no rename)
const folderMoves = new Map(); // groupIdx -> path of file whose folder to move into (null = no move)
const $id = id => document.getElementById(id);


async function init() {
  try {
    const resp = await fetch('/groups');
    groups = await resp.json();
  } catch (e) {
    $id('groups-container').textContent = `Failed to load groups: ${e}`;
    return;
  }
  groups.forEach((g, gi) => {
    selections.set(gi, new Set([g.keep_default]));
    if (g.rename_src !== null) renames.set(gi, g.rename_src);
    if (g.folder_src !== null) folderMoves.set(gi, g.folder_src);
  });
  updateStats();
  renderPage();
}

function renderPage() {
  const container = $id('groups-container');
  container.innerHTML = '';
  const start = currentPage * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, groups.length);
  for (let gi = start; gi < end; gi++) {
    container.appendChild(renderGroup(gi, groups[gi]));
    refreshHints(gi);
  }
  const total = Math.max(1, Math.ceil(groups.length / PAGE_SIZE));
  $id('page-info').textContent = `Page ${currentPage + 1} of ${total}  (${groups.length} groups total)`;
  $id('prev-btn').disabled = currentPage === 0;
  $id('next-btn').disabled = currentPage >= total - 1;
  window.scrollTo(0, 0);
}

function renderGroup(gi, group) {
  const card = document.createElement('div');
  card.className = 'group-card';
  card.id = `group-${gi}`;

  const header = document.createElement('div');
  header.className = 'group-header';
  header.textContent = `Group ${gi + 1}  \u00b7  ${group.files.length} files`;
  card.appendChild(header);

  const slots = document.createElement('div');
  slots.className = 'file-slots';

  group.files.forEach((f, fi) => {
    const slot = document.createElement('div');
    slot.className = 'file-slot';
    slot.id = `slot-${gi}-${fi}`;

    const kept = selections.get(gi)?.has(f.path);
    if (!kept) slot.classList.add('deselected');

    if (fi === 0) {
      const badge = document.createElement('div');
      badge.className = 'original-badge';
      badge.textContent = 'Likely original';
      slot.appendChild(badge);
    }

    if (f.is_video) {
      // Clickable thumbnail that opens the video for playback
      const link = document.createElement('a');
      link.href = `/image?path=${encodeURIComponent(f.path)}`;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';

      const img = document.createElement('img');
      img.loading = 'lazy';
      img.src = `/thumbnail?path=${encodeURIComponent(f.path)}`;
      img.alt = f.name;
      img.title = f.path + ' (click to play)';
      img.onerror = function() {
        const ph = document.createElement('div');
        ph.className = 'img-placeholder';
        ph.textContent = f.name;
        link.replaceChild(ph, img);
      };
      link.appendChild(img);
      slot.appendChild(link);
    } else {
      // Clickable image link
      const link = document.createElement('a');
      link.href = `/image?path=${encodeURIComponent(f.path)}`;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';

      const img = document.createElement('img');
      img.loading = 'lazy';
      img.src = `/image?path=${encodeURIComponent(f.path)}`;
      img.alt = f.name;
      img.title = f.path;
      img.onerror = function() {
        const ph = document.createElement('div');
        ph.className = 'img-placeholder';
        ph.textContent = f.name;
        link.replaceChild(ph, img);
      };
      img.addEventListener('mouseenter', () => showPreview(img.src, img));
      img.addEventListener('mouseleave', hidePreview);
      link.appendChild(img);
      slot.appendChild(link);
    }

    // File info
    const info = document.createElement('div');
    info.className = 'slot-info';

    const nameEl = document.createElement('div');
    nameEl.className = 'slot-name';
    nameEl.textContent = f.name;

    const pathEl = document.createElement('div');
    pathEl.className = 'slot-path';
    pathEl.textContent = f.rel;
    pathEl.title = f.path;

    const sizeEl = document.createElement('div');
    sizeEl.className = 'slot-size';
    sizeEl.textContent = f.size_fmt;

    const keepLabel = document.createElement('label');
    keepLabel.className = 'slot-keep';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!kept;
    cb.addEventListener('change', () => toggleKeep(gi, f.path, cb.checked));
    keepLabel.appendChild(cb);
    keepLabel.appendChild(document.createTextNode(' Keep'));

    const useNameBtn = document.createElement('button');
    useNameBtn.className = 'slot-use-name-btn';
    useNameBtn.id = `rename-btn-${gi}-${fi}`;
    const isRenameActive = renames.get(gi) === f.path;
    useNameBtn.classList.toggle('active', isRenameActive);
    useNameBtn.textContent = isRenameActive ? '\u2713 Name source' : 'Use this name';
    useNameBtn.disabled = !!selections.get(gi)?.has(f.path);
    useNameBtn.addEventListener('click', () => setRenameSource(gi, f.path));

    const renameHint = document.createElement('div');
    renameHint.className = 'slot-rename-hint';
    renameHint.id = `rename-hint-${gi}-${fi}`;

    const useFolderBtn = document.createElement('button');
    useFolderBtn.className = 'slot-use-folder-btn';
    useFolderBtn.id = `folder-btn-${gi}-${fi}`;
    const isFolderActive = folderMoves.get(gi) === f.path;
    useFolderBtn.classList.toggle('active', isFolderActive);
    useFolderBtn.textContent = isFolderActive ? '\u2713 Folder source' : 'Use this folder';
    useFolderBtn.disabled = !!selections.get(gi)?.has(f.path);
    useFolderBtn.addEventListener('click', () => setFolderSource(gi, f.path));

    const folderHint = document.createElement('div');
    folderHint.className = 'slot-folder-hint';
    folderHint.id = `folder-hint-${gi}-${fi}`;

    info.appendChild(nameEl);
    info.appendChild(pathEl);
    info.appendChild(sizeEl);
    info.appendChild(keepLabel);
    info.appendChild(useNameBtn);
    info.appendChild(renameHint);
    info.appendChild(useFolderBtn);
    info.appendChild(folderHint);
    slot.appendChild(info);
    slots.appendChild(slot);
  });

  card.appendChild(slots);
  return card;
}

function toggleKeep(gi, path, checked) {
  let sel = selections.get(gi);
  if (!sel) { sel = new Set(); selections.set(gi, sel); }
  if (checked) sel.add(path); else sel.delete(path);

  const group = groups[gi];
  // If a file that was the rename source is now being kept, auto-clear it
  const renameSrc = renames.get(gi);
  if (renameSrc && sel.has(renameSrc)) {
    renames.delete(gi);
    group.files.forEach((f, fi) => {
      const btn = $id(`rename-btn-${gi}-${fi}`);
      if (btn) { btn.classList.remove('active'); btn.textContent = 'Use this name'; }
    });
  }

  // If a file that was the folder source is now being kept, auto-clear it
  const folderSrc = folderMoves.get(gi);
  if (folderSrc && sel.has(folderSrc)) {
    folderMoves.delete(gi);
    group.files.forEach((f, fi) => {
      const btn = $id(`folder-btn-${gi}-${fi}`);
      if (btn) { btn.classList.remove('active'); btn.textContent = 'Use this folder'; }
    });
  }

  // Refresh visual state for all slots in this group
  group.files.forEach((f, fi) => {
    const slot = $id(`slot-${gi}-${fi}`);
    if (slot) {
      if (sel.has(f.path)) slot.classList.remove('deselected');
      else slot.classList.add('deselected');
    }
  });
  refreshHints(gi);
  updateStats();
}

function setRenameSource(gi, path) {
  if (renames.get(gi) === path) {
    renames.delete(gi);
  } else {
    renames.set(gi, path);
  }
  const group = groups[gi];
  group.files.forEach((f, fi) => {
    const btn = $id(`rename-btn-${gi}-${fi}`);
    if (btn) {
      const isActive = renames.get(gi) === f.path;
      btn.classList.toggle('active', isActive);
      btn.textContent = isActive ? '\u2713 Name source' : 'Use this name';
    }
  });
  refreshHints(gi);
}

function setFolderSource(gi, path) {
  if (folderMoves.get(gi) === path) {
    folderMoves.delete(gi);
  } else {
    folderMoves.set(gi, path);
  }
  const group = groups[gi];
  group.files.forEach((f, fi) => {
    const btn = $id(`folder-btn-${gi}-${fi}`);
    if (btn) {
      const isActive = folderMoves.get(gi) === f.path;
      btn.classList.toggle('active', isActive);
      btn.textContent = isActive ? '\u2713 Folder source' : 'Use this folder';
    }
  });
  refreshHints(gi);
}

function refreshHints(gi) {
  const renameSrc = renames.get(gi) || null;
  const folderSrc = folderMoves.get(gi) || null;
  const kept = selections.get(gi) || new Set();
  const keptArr = [...kept];
  const singleKept = keptArr.length === 1;
  // Both transforms apply only when exactly one file is kept and the source is not that file
  const willRename = !!(renameSrc && singleKept && !kept.has(renameSrc));
  const willMoveFolder = !!(folderSrc && singleKept && !kept.has(folderSrc));
  const newName = willRename
    ? (groups[gi].files.find(f => f.path === renameSrc) || {}).name
    : null;
  const targetDirRel = willMoveFolder
    ? (groups[gi].files.find(f => f.path === folderSrc) || {}).dir_rel
    : null;
  const keptPath = singleKept ? keptArr[0] : null;
  groups[gi].files.forEach((f, fi) => {
    const renameHint = $id(`rename-hint-${gi}-${fi}`);
    if (renameHint) {
      renameHint.textContent = (keptPath === f.path && newName)
        ? `\u2192 Will be renamed to: ${newName}` : '';
    }
    const folderHint = $id(`folder-hint-${gi}-${fi}`);
    if (folderHint) {
      folderHint.textContent = (keptPath === f.path && targetDirRel)
        ? `\u2192 Will be moved to: ${targetDirRel}` : '';
    }
    // Disable both action buttons on kept files (conflict prevention)
    const nameBtn = $id(`rename-btn-${gi}-${fi}`);
    if (nameBtn) nameBtn.disabled = kept.has(f.path);
    const folderBtn = $id(`folder-btn-${gi}-${fi}`);
    if (folderBtn) folderBtn.disabled = kept.has(f.path);
  });
}

function updateStats() {
  let delFiles = 0, delBytes = 0;
  groups.forEach((g, gi) => {
    const kept = selections.get(gi) || new Set();
    g.files.forEach(f => {
      if (!kept.has(f.path)) { delFiles++; delBytes += f.size; }
    });
  });
  const mb = (delBytes / (1024 * 1024)).toFixed(1);
  $id('stats').textContent =
    `${delFiles} file${delFiles !== 1 ? 's' : ''} to delete \u00b7 ${mb} MB recoverable`;
}

function changePage(delta) {
  const total = Math.max(1, Math.ceil(groups.length / PAGE_SIZE));
  currentPage = Math.max(0, Math.min(currentPage + delta, total - 1));
  renderPage();
}

async function confirmSelections() {
  // Validate: at least 1 file kept per group
  for (let gi = 0; gi < groups.length; gi++) {
    const kept = selections.get(gi) || new Set();
    if (kept.size === 0) {
      const page = Math.floor(gi / PAGE_SIZE);
      if (page !== currentPage) { currentPage = page; renderPage(); }
      const card = $id(`group-${gi}`);
      if (card) card.scrollIntoView({ behavior: 'smooth' });
      alert(`Group ${gi + 1}: please keep at least one file before confirming.`);
      return;
    }
  }

  const keepPaths = [];
  selections.forEach(set => set.forEach(p => keepPaths.push(p)));

  const transformList = [];
  for (let gi = 0; gi < groups.length; gi++) {
    const kept = selections.get(gi) || new Set();
    const keptArr = [...kept];
    if (keptArr.length !== 1) continue;
    const keptPath = keptArr[0];
    const renameSrc = renames.get(gi);
    const folderSrc = folderMoves.get(gi);
    const willRename = renameSrc && !kept.has(renameSrc);
    const willMove = folderSrc && !kept.has(folderSrc);
    if (!willRename && !willMove) continue;
    const entry = {path: keptPath};
    if (willRename) {
      const f = groups[gi].files.find(g => g.path === renameSrc);
      if (f) entry.new_name = f.name;
    }
    if (willMove) {
      const f = groups[gi].files.find(g => g.path === folderSrc);
      if (f) entry.target_dir = f.dir;
    }
    transformList.push(entry);
  }

  $id('confirm-btn').disabled = true;
  let result;
  try {
    const resp = await fetch('/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keep: keepPaths, renames: transformList }),
    });
    result = await resp.json();
  } catch (e) {
    alert(`Error communicating with server: ${e}`);
    $id('confirm-btn').disabled = false;
    return;
  }
  showSummary(result);
}

function showSummary(result) {
  $id('dialog-title').textContent = 'Review complete';
  $id('dialog-moved').textContent = `\u2705 Moved to trash: ${result.moved} file(s)`;
  $id('dialog-renamed').textContent =
    result.renamed > 0 ? `\u270f Moved/renamed: ${result.renamed} file(s)` : '';
  $id('dialog-skipped').textContent =
    result.skipped > 0 ? `\u26a0\ufe0f Skipped (already gone): ${result.skipped}` : '';
  $id('dialog-errors').textContent =
    result.errors?.length ? `\u274c Errors: ${result.errors.join('; ')}` : '';
  $id('overlay').classList.add('active');
}

function showPreview(imgSrc, el) {
  const pv = $id('hover-preview');
  const pvImg = $id('hover-preview-img');
  pvImg.src = imgSrc;
  const rect = el.getBoundingClientRect();
  const GAP = 14, M = 8;
  // Measure actual space available on each side of the thumbnail, then take
  // whichever side is larger and cap at 1024. This fills the available room
  // rather than applying a blanket viewport-fraction that was too restrictive.
  const spaceRight = window.innerWidth  - rect.right - GAP - M;
  const spaceLeft  = rect.left - GAP - M;
  const spaceH     = window.innerHeight - 2 * M;
  const maxSide = Math.min(1024, Math.max(spaceRight, spaceLeft), spaceH);
  pvImg.style.maxWidth  = `${maxSide}px`;
  pvImg.style.maxHeight = `${maxSide}px`;
  pv.style.display = 'block';
  let left = rect.right + GAP;
  if (left + maxSide > window.innerWidth - M) { left = rect.left - maxSide - GAP; }
  if (left < M) { left = M; }
  let top = rect.top + rect.height / 2 - maxSide / 2;
  if (top < M) { top = M; }
  if (top + maxSide > window.innerHeight - M) { top = window.innerHeight - maxSide - M; }
  pv.style.left = `${left}px`;
  pv.style.top  = `${top}px`;
}

function hidePreview() {
  $id('hover-preview').style.display = 'none';
}

async function doneProceed() {
  $id('done-btn').disabled = true;
  $id('dialog-title').textContent = 'Done';
  $id('dialog-moved').textContent = 'You may close this tab.';
  $id('dialog-renamed').textContent = '';
  $id('dialog-skipped').textContent = '';
  $id('dialog-errors').textContent = '';
  try { await fetch('/shutdown', { method: 'POST' }); } catch (_) {}
}

init();
</script>
</body>
</html>"""
# The sentinel uses JS block-comment syntax so the literal '10' is valid
# JavaScript and the string can be located unambiguously for substitution at
# runtime (launch_review_ui replaces it with the actual --page-size value).
assert "/*PAGESIZE*/10/*PAGESIZE*/" in _REVIEW_HTML, (
    "BUG: _REVIEW_HTML is missing the /*PAGESIZE*/10/*PAGESIZE*/ placeholder"
)


def _resolve_transform(r: dict, keep_set: set, dir_resolved: Path):
    """Validate and resolve one rename/move entry.

    Returns (src, dst) on success, None for a no-op, or an error string.
    """
    from_path  = r.get("path")   if isinstance(r.get("path"),       str) else ""
    new_name   = r.get("new_name")   or ""
    target_dir = r.get("target_dir") or ""
    if not isinstance(new_name,   str): new_name   = ""
    if not isinstance(target_dir, str): target_dir = ""

    if from_path not in keep_set:
        return f"Cannot transform non-kept file: {os.path.basename(from_path)}"
    try:
        src = Path(from_path).resolve()
        src.relative_to(dir_resolved)
    except (ValueError, RuntimeError):
        return f"Invalid path: {os.path.basename(from_path)}"
    if not src.is_file():
        return f"File not found: {os.path.basename(from_path)}"

    if new_name:
        safe_name = Path(new_name).name
        if not safe_name:
            return f"Invalid new name: {new_name!r}"
    else:
        safe_name = src.name

    if target_dir:
        try:
            dst_dir = Path(target_dir).resolve()
            dst_dir.relative_to(dir_resolved)
        except (ValueError, RuntimeError):
            return f"Invalid target folder for {src.name}"
        if not dst_dir.is_dir():
            return f"Target folder not found for {src.name}"
    else:
        dst_dir = src.parent

    dst = dst_dir / safe_name
    if dst == src:
        return None  # no-op
    if dst.exists():
        return f"Cannot move/rename '{safe_name}': already exists in target folder"
    return src, dst


def launch_review_ui(groups: list, directory: str, page_size: int = 10,
                     ignore: list | None = None) -> None:
    """Start a local Flask server and open a browser-based duplicate review UI."""
    if not groups:
        print("\nNo duplicate groups found. Nothing to review.")
        return

    try:
        from flask import Flask, jsonify, request, send_file, abort, Response
    except ImportError:
        print("Error: Flask is required for --review.", file=sys.stderr)
        print("  pip install flask", file=sys.stderr)
        sys.exit(1)

    import logging
    import threading
    import time
    import webbrowser

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    import flask.cli
    flask.cli.show_server_banner = lambda *_a, **_kw: None

    app = Flask(__name__)
    # Sort groups globally by their first folder so same-folder duplicates
    # cluster together. Files within each group keep their size-descending order
    # (largest = likely original) as established by group_duplicates.
    sorted_groups = sorted(groups, key=lambda g: str(Path(g[0][0]).parent))
    dir_resolved = Path(directory).resolve()
    extra = tuple(sorted(p.lower() for p in (ignore or [])))
    groups_data = _groups_to_json(sorted_groups, directory, extra, str(dir_resolved))
    shutdown_event = threading.Event()

    html = _REVIEW_HTML.replace(
        "/*PAGESIZE*/10/*PAGESIZE*/", str(page_size), 1
    )

    @app.route("/")
    def index():
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/groups")
    def get_groups():
        return jsonify(groups_data)

    @app.route("/image")
    def image():
        raw = request.args.get("path", "")
        try:
            resolved = Path(raw).resolve()
            resolved.relative_to(dir_resolved)  # raises ValueError if outside
        except (ValueError, RuntimeError):
            abort(403)
        if not resolved.is_file():
            abort(404)
        try:
            return send_file(str(resolved))
        except Exception:
            abort(404)

    @app.route("/thumbnail")
    def thumbnail():
        import subprocess
        raw = request.args.get("path", "")
        try:
            resolved = Path(raw).resolve()
            resolved.relative_to(dir_resolved)
        except (ValueError, RuntimeError):
            abort(403)
        if not resolved.is_file():
            abort(404)
        for seek in ("1", "0"):
            try:
                r = subprocess.run(
                    ["ffmpeg", "-nostdin", "-ss", seek, "-i", str(resolved),
                     "-vframes", "1", "-f", "image2", "-vcodec", "mjpeg", "pipe:1",
                     "-loglevel", "error"],
                    capture_output=True, timeout=15,
                )
                if r.stdout:
                    return Response(r.stdout, mimetype="image/jpeg")
            except Exception:
                pass
        abort(404)

    @app.route("/confirm", methods=["POST"])
    def confirm():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request body"}), 400
        # Accept only string paths; non-strings are silently dropped so a
        # malformed keep list can never cause all files to be trashed.
        keep_set = {p for p in data.get("keep", []) if isinstance(p, str)}
        # Server-side guard: every group must keep at least one file.
        for g in groups_data:
            if not any(f["path"] in keep_set for f in g["files"]):
                return jsonify({"error": "Each group must keep at least one file"}), 400
        all_paths = {f["path"] for g in groups_data for f in g["files"]}
        to_delete = all_paths - keep_set
        renamed = 0
        moved = 0
        skipped = 0
        errors = []
        # Trash first, then transform.
        #
        # Order matters: if a kept file's rename/move target happens to be the
        # same path as a to-be-trashed file (e.g. keep photos/A/img.jpg and
        # move it into photos/B/ where photos/B/img.jpg is also being trashed),
        # doing the trash first vacates that path so the transform can succeed.
        # Doing transforms first would falsely report a collision and then trash
        # the very file we just moved there, silently destroying the kept copy.
        for path in to_delete:
            if not os.path.exists(path):
                skipped += 1
                continue
            try:
                dst = _unique_dst(_trash_path(path, str(dir_resolved)))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, str(dst))
                moved += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        # Apply renames/moves to kept files after the trash step has cleared
        # any conflicting paths.
        renames_in = data.get("renames", [])
        if isinstance(renames_in, list):
            for r in renames_in:
                if not isinstance(r, dict):
                    continue
                result = _resolve_transform(r, keep_set, dir_resolved)
                if result is None:
                    continue  # no-op
                if isinstance(result, str):
                    errors.append(result)
                    continue
                src, dst = result
                try:
                    src.rename(dst)
                    renamed += 1
                except Exception as e:
                    errors.append(f"Move/rename failed for {src.name}: {e}")
        return jsonify({"moved": moved, "skipped": skipped, "renamed": renamed, "errors": errors})

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        shutdown_event.set()
        return "", 204

    import socket
    import urllib.request

    # Find a free port in range 5757–5766.
    port = None
    for _candidate in range(5757, 5767):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            try:
                _s.bind(("127.0.0.1", _candidate))
                port = _candidate
                break
            except OSError:
                continue
    if port is None:
        print("Error: no free port available in range 5757–5766.", file=sys.stderr)
        return

    def _run():
        try:
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        except OSError as e:
            print(f"Error: could not start server on port {port}: {e}", file=sys.stderr)
            shutdown_event.set()

    server_ready = threading.Event()

    def _poll_server():
        for _ in range(20):
            if shutdown_event.is_set():
                return
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/groups", timeout=0.5
                )
                server_ready.set()
                return
            except Exception:
                time.sleep(0.5)

    threading.Thread(target=_run, daemon=True).start()
    threading.Thread(target=_poll_server, daemon=True).start()
    if not server_ready.wait(timeout=10) or shutdown_event.is_set():
        if not shutdown_event.is_set():
            print(f"Warning: server may not be ready; opening browser anyway.",
                  file=sys.stderr)
        else:
            return
    webbrowser.open(f"http://127.0.0.1:{port}/")
    print(f"\nReview UI at http://127.0.0.1:{port}/  - confirm in browser, then click Done.")
    try:
        while not shutdown_event.wait(timeout=1):
            pass
    except KeyboardInterrupt:
        print("\nInterrupted.")
    print("Review complete.")

# ── Interactive CLI mode ───────────────────────────────────────────────────────


def interactive_mode(groups: list, directory: str, ignore: tuple = ()) -> None:
    """Step through duplicate groups interactively.

    For each group the best filename is chosen automatically; the user picks
    which directory to keep the file in by entering a letter (a, b, c …).
    All other replicas are moved to __duplicate_files_trash/.  Enter 's' to skip a group.
    """
    if not groups:
        print("\nNo duplicate groups found.")
        return

    LETTERS = "abcdefghijklmnopqrstuvwxyz"
    total = len(groups)
    moved_count = renamed_count = skipped_count = error_count = 0
    skip_all = False

    # Sort groups: primary = lexicographically first folder any replica is in,
    # secondary = sorted filenames of replicas inside that folder.
    def _group_sort_key(g):
        first_folder = min(str(Path(p).parent) for p, _ in g)
        names = sorted(os.path.basename(p) for p, _ in g if str(Path(p).parent) == first_folder)
        return (first_folder, names)
    groups = sorted(groups, key=_group_sort_key)

    print(f"\n{total} duplicate group(s) to review interactively.")
    print("Enter a letter to pick the directory, 's' to skip, or 'x' to skip all remaining.\n")

    for group_idx, group in enumerate(groups, 1):
        files_meta = [
            {"path": p, "name": os.path.basename(p),
             "dir": str(Path(p).parent), "size": sz}
            for p, sz in group
        ]

        # Auto-select best filename
        best_name = files_meta[0]["name"]
        best_ns = _name_score(best_name, ignore)
        for f in files_meta[1:]:
            ns = _name_score(f["name"], ignore)
            if ns > best_ns:
                best_ns, best_name = ns, f["name"]

        # Ordered unique directories (first-seen order)
        seen_dirs: list = []
        seen_dirs_set: set = set()
        for f in files_meta:
            d = f["dir"]
            if d not in seen_dirs_set:
                seen_dirs_set.add(d)
                seen_dirs.append(d)

        n_dirs = min(len(seen_dirs), len(LETTERS))
        same_folder = len(seen_dirs) == 1

        if skip_all:
            skipped_count += 1
            continue

        if same_folder:
            # All replicas are in the same directory, so there is no folder to
            # choose between.  Show the auto-selected keeper (scored by filename)
            # and let the user confirm ('k') or skip ('s').
            preview_keep_path = _pick_keeper(files_meta, ignore, directory)
            link = Path(files_meta[0]["path"]).as_uri()
            print(f"{best_name}  {link}")
            print(f"  Same folder: {seen_dirs[0]}")
            for f in files_meta:
                label = "keep " if f["path"] == preview_keep_path else "trash"
                print(f"  {label}  {f['name']}")

            answer = None
            while answer not in {"k", "s", "x"}:
                try:
                    raw = input("  [k=keep best/s=skip/x=skip all]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    print(f"\nSummary: {moved_count} trashed, {renamed_count} renamed, "
                          f"{skipped_count} skipped, {error_count} error(s).")
                    return
                if raw in {"k", "s", "x"}:
                    answer = raw

            print()

            if answer == "x":
                skip_all = True
                skipped_count += 1
                continue

            if answer == "s":
                skipped_count += 1
                continue

            chosen_dir = seen_dirs[0]
        else:
            link = Path(files_meta[0]["path"]).as_uri()
            print(f"{best_name}  {link}")
            for i in range(n_dirs):
                print(f"  {LETTERS[i]}) {_safe_relpath(seen_dirs[i], directory)}")

            valid = set(LETTERS[:n_dirs]) | {"s", "x"}
            prompt = f"  [{'/'.join(LETTERS[:n_dirs])}/s=skip/x=skip all]: "
            answer = None
            while answer not in valid:
                try:
                    raw = input(prompt).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    print(f"\nSummary: {moved_count} trashed, {renamed_count} renamed, "
                          f"{skipped_count} skipped, {error_count} error(s).")
                    return
                if raw in valid:
                    answer = raw

            print()

            if answer == "x":
                skip_all = True
                skipped_count += 1
                continue

            if answer == "s":
                skipped_count += 1
                continue

            chosen_dir = seen_dirs[LETTERS.index(answer)]

        # Split into files to keep (in chosen_dir) and files to trash
        keep_candidates = [f for f in files_meta if f["dir"] == chosen_dir]
        trash_files = [f for f in files_meta if f["dir"] != chosen_dir]

        # If multiple files landed in chosen_dir, keep the best-scored one
        if len(keep_candidates) > 1:
            keeper = keep_candidates[0]
            keeper_score = (_folder_score(keeper["dir"], ignore, directory),
                            _name_score(keeper["name"], ignore))
            for f in keep_candidates[1:]:
                s = (_folder_score(f["dir"], ignore, directory),
                     _name_score(f["name"], ignore))
                if s > keeper_score:
                    trash_files.append(keeper)
                    keeper_score, keeper = s, f
                else:
                    trash_files.append(f)
        else:
            keeper = keep_candidates[0]

        # Trash non-kept replicas
        for f in trash_files:
            p = f["path"]
            if not os.path.exists(p):
                skipped_count += 1
                continue
            try:
                try:
                    dst = _unique_dst(_trash_path(p, directory))
                except ValueError:
                    # File is outside directory (e.g. symlink target); fall back
                    # to a flat path inside the trash folder.
                    dst = _unique_dst(
                        Path(directory) / "__duplicate_files_trash" / Path(p).name
                    )
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, str(dst))
                moved_count += 1
                print(f"  trashed  {_safe_relpath(p, directory)}")
            except Exception as e:
                print(f"  Warning: could not trash {os.path.basename(p)}: {e}")
                error_count += 1

        # Rename kept file to best_name if it differs
        kept_path = Path(keeper["path"])
        if kept_path.name != best_name:
            new_path = kept_path.parent / best_name
            if new_path.exists():
                print(f"  Note: cannot rename to '{best_name}' "
                      f"(already exists); keeping '{kept_path.name}'")
            else:
                try:
                    kept_path.rename(new_path)
                    print(f"  renamed  {kept_path.name} -> {best_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"  Warning: could not rename to '{best_name}': {e}")
                    error_count += 1

        print()

    print(f"Done. {moved_count} file(s) trashed, {renamed_count} renamed, "
          f"{skipped_count} skipped, {error_count} error(s).")


# ── Trash verification ─────────────────────────────────────────────────────────

def verify_trash(directory: str, threshold: int, cache_dir: str | None = None) -> None:
    """Verify every media file in __duplicate_files_trash/ has a match in directory.

    Images are matched perceptually (phash+dhash within *threshold*).
    Videos are matched by exact checksum.
    Exits with code 1 if any trash file has no match.
    """
    trash_dir = os.path.join(directory, "__duplicate_files_trash")
    if not os.path.isdir(trash_dir):
        print(f"__duplicate_files_trash/ not found in {directory}.")
        return
    src_cache_dir = cache_dir if cache_dir else _ensure_db_dir(directory)

    # 1. Scan trash — split into images and videos
    print(f"Scanning trash: {trash_dir} ...")
    trash_all = scan_files(trash_dir, MEDIA_EXTENSIONS)
    trash_img_files = [f for f in trash_all if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
    trash_vid_files = [f for f in trash_all if Path(f).suffix.lower() in VIDEO_EXTENSIONS]
    print(f"  {len(trash_img_files)} images  *  {len(trash_vid_files)} videos in trash")

    if not trash_all:
        print("Trash is empty — nothing to verify.")
        return

    unmatched: list = []

    # ── Image verification (phash/dhash) ──────────────────────────────────────
    if trash_img_files:
        print(f"\nScanning {directory} for images ...")
        src_img_files = scan_files(directory)  # excludes __duplicate_files_trash/ automatically
        print(f"  {len(src_img_files)} images in source")

        cache_path = os.path.join(src_cache_dir, CACHE_FILENAME)
        cache = load_cache(cache_path)
        live_keys = {path_without_drive(p) for p in src_img_files}
        cache = {k: v for k, v in cache.items() if k in live_keys}

        print(f"\nHashing source images ...")
        cache_out = open_cache_for_append(cache_path)
        try:
            src_hashes, new_count, _rehashed, errors = build_hashes(
                src_img_files, cache, cache_out)
        finally:
            cache_out.close()
        save_cache(cache_path, cache)
        print(f"  {len(src_hashes)} hashed  *  {new_count} new  *  {errors} errors")

        trash_cache_path = os.path.join(_ensure_db_dir(trash_dir), CACHE_FILENAME)
        trash_cache = load_cache(trash_cache_path)
        trash_img_keys = {path_without_drive(p) for p in trash_img_files}
        trash_cache = {k: v for k, v in trash_cache.items() if k in trash_img_keys}

        print(f"\nHashing trash images ...")
        trash_cache_out = open_cache_for_append(trash_cache_path)
        try:
            trash_img_hashes, new_count, _rehashed, err_count = build_hashes(
                trash_img_files, trash_cache, trash_cache_out)
        finally:
            trash_cache_out.close()
        save_cache(trash_cache_path, trash_cache)
        print(f"  {len(trash_img_hashes)} hashed  *  {new_count} new  *  {err_count} errors")

        print(f"\nVerifying images (threshold={threshold}) ...")
        if src_hashes:
            def hamming(a, b):
                return (a[1] ^ b[1]).bit_count()

            src_items = [(p, ph) for p, (ph, _dh, _sz) in src_hashes.items()]
            tree = pybktree.BKTree(hamming, src_items)

            n = len(trash_img_hashes)
            for i, (path, (ph, dh, _sz)) in enumerate(trash_img_hashes.items(), 1):
                if i % 50 == 0 or i == n:
                    _progress_bar(i, n, os.path.basename(path))
                matches = tree.find(("", ph), threshold)
                good = [m for m in matches
                        if (src_hashes[m[1][0]][1] ^ dh).bit_count() <= threshold]
                if not good:
                    unmatched.append(path)
            print()
        else:
            unmatched.extend(trash_img_hashes.keys())

    # ── Video verification (exact checksum) ───────────────────────────────────
    if trash_vid_files:
        print(f"\nScanning {directory} for videos ...")
        src_vid_files = scan_files(directory, VIDEO_EXTENSIONS)
        print(f"  {len(src_vid_files)} videos in source")

        # Additive update: add video checksums to source db without removing
        # any image entries that may already be present.
        src_db_path = os.path.join(src_cache_dir, DB_FILENAME)
        print(f"\nIndexing source videos ...")
        src_db, src_new, src_errors = _update_checksums_additive(src_vid_files, src_db_path)
        src_hits = len(src_vid_files) - src_new
        print(f"  {len(src_vid_files)} indexed  *  {src_new} new  *  "
              f"{src_hits} hits  *  {src_errors} errors")

        src_vid_keys = {path_without_drive(p) for p in src_vid_files}
        src_checksums = {entry["checksum"]
                         for k, entry in src_db.items() if k in src_vid_keys}

        # build_exact_index is safe for the trash db: it's video-only and we
        # want gone-file pruning so stale entries don't linger.
        trash_db_path = os.path.join(_ensure_db_dir(trash_dir), DB_FILENAME)
        print(f"\nIndexing trash videos ...")
        trash_db, trash_new, gone_count, trash_errors = build_exact_index(
            trash_vid_files, trash_db_path)
        trash_hits = len(trash_db) - trash_new
        print(f"  {len(trash_db)} indexed  *  {trash_new} new  *  "
              f"{trash_hits} hits  *  {trash_errors} errors  *  {gone_count} pruned")

        print(f"\nVerifying videos ...")
        abs_path_map = {path_without_drive(p): p for p in trash_vid_files}
        n = len(trash_db)
        exact_unmatched_vids = []
        for i, (k, entry) in enumerate(trash_db.items(), 1):
            if i % 50 == 0 or i == n:
                _progress_bar(i, n, os.path.basename(entry["path"]))
            if entry["checksum"] not in src_checksums:
                exact_unmatched_vids.append(abs_path_map.get(k, k))
        print()

        # Perceptual fallback: if a video was kept and later re-encoded (e.g. at a
        # different resolution or bitrate), its byte content changes so the exact
        # checksum no longer matches the trashed original.  Frame-level pHash
        # comparison catches these cases — the same visual content hashes similarly
        # even across re-encodes that change every byte in the file.
        if exact_unmatched_vids and ffmpeg_available():
            print(f"\nPerceptual check for {len(exact_unmatched_vids)} "
                  f"unmatched video(s) (threshold={threshold}) ...")

            src_vcache_path = os.path.join(src_cache_dir, VCACHE_FILENAME)
            src_vcache = load_vcache(src_vcache_path)
            src_vhash_out = _open_vcache_for_append(src_vcache_path)
            try:
                src_vhashes, _, _, _ = build_video_hashes(
                    src_vid_files, src_vcache, src_vhash_out)
            finally:
                src_vhash_out.close()
            save_vcache(src_vcache_path, src_vcache)

            trash_vcache_path = os.path.join(_ensure_db_dir(trash_dir), VCACHE_FILENAME)
            trash_vcache = load_vcache(trash_vcache_path)
            trash_vhash_out = _open_vcache_for_append(trash_vcache_path)
            try:
                trash_vhashes, _, _, _ = build_video_hashes(
                    exact_unmatched_vids, trash_vcache, trash_vhash_out)
            finally:
                trash_vhash_out.close()
            save_vcache(trash_vcache_path, trash_vcache)

            for path, (dur_t, frames_t, _) in trash_vhashes.items():
                matched = False
                for _src, (dur_s, frames_s, _) in src_vhashes.items():
                    tol = max(10.0, 0.05 * max(dur_t, dur_s))
                    if abs(dur_t - dur_s) > tol:
                        continue
                    if _video_distance(frames_t, frames_s) <= threshold:
                        matched = True
                        break
                if not matched:
                    unmatched.append(path)
        else:
            unmatched.extend(exact_unmatched_vids)

    # ── Report ────────────────────────────────────────────────────────────────
    total_checked = len(trash_img_files) + len(trash_vid_files)
    print(f"\nVerification: {total_checked} trash file(s) checked")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} file(s) have no match in source "
              f"(may have been wrongly trashed):\n")
        for p in sorted(unmatched):
            print(f"    {_fmt_path(p)}")
        sys.exit(1)
    else:
        print("  OK — all trash files have a match in source.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate images and videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0].strip(),
    )
    parser.add_argument("directory", help="Directory to scan recursively")
    parser.add_argument(
        "--exact", action="store_true",
        help=(
            "Use SHA-256 checksums for exact duplicate detection. "
            "Covers all media files including videos. Reads and updates "
            f"{DB_DIR}/{DB_FILENAME} in the scanned directory. Ignores --threshold."
        ),
    )
    parser.add_argument(
        "--video", action="store_true",
        help=(
            "Perceptual duplicate detection for video files. Samples "
            f"{N_VIDEO_FRAMES} frames per video with ffmpeg and compares mean "
            "pHash Hamming distance, so re-encodes at different resolutions or "
            "bitrates are matched. Requires ffmpeg on PATH. Can be combined with "
            f"the default image scan. Frame hashes cached in {DB_DIR}/{VCACHE_FILENAME}. "
            "Ignored when --exact is used."
        ),
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=DEFAULT_THRESHOLD, metavar="N",
        help=f"Hamming distance threshold for perceptual mode (default: {DEFAULT_THRESHOLD}). "
             f"Ignored with --exact.",
    )
    parser.add_argument(
        "--cache", "-c", default=None, metavar="DIR",
        help=f"Directory in which to store hash database files "
             f"(default: <scanned-dir>/{DB_DIR}/). "
             f"Applies to all modes, including --exact.",
    )
    parser.add_argument(
        "--delta", "-d", default="0", metavar="SIZE",
        help=(
            "Only report groups where the file-size spread (largest minus smallest) "
            "is at least SIZE. Accepts optional suffix: kb, mb, gb (case-insensitive). "
            "Examples: 100, 50kb, 2mb. Default: 0 (no filter). "
            "Note: exact duplicates always share the same size, so this filter "
            "has no effect in --exact mode."
        ),
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--review", action="store_true",
        help=(
            "Launch a local browser UI to interactively review duplicates and move "
            "unwanted files to __duplicate_files_trash/ inside the scanned directory. "
            "Requires Flask (pip install flask)."
        ),
    )
    output_group.add_argument(
        "--interactive", action="store_true",
        help=(
            "Step through each duplicate group in the terminal. The best filename "
            "is chosen automatically; you pick the directory to keep it in (a/b/c…) "
            "or press 's' to skip. Other replicas are moved to __duplicate_files_trash/."
        ),
    )
    parser.add_argument(
        "--page-size", "-p", type=int, default=10, metavar="N",
        help="Number of duplicate groups shown per page in --review mode (1-500, default: 10).",
    )
    parser.add_argument(
        "--ignore", "-i", action="append", default=[], metavar="WORD",
        help=(
            "Treat WORD as noise when scoring file and folder names in --review and --interactive modes, "
            "so it never influences which name or folder is auto-selected. "
            "Case-insensitive. Repeatable: --ignore vacation --ignore beach."
        ),
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, metavar="N",
        help="Maximum number of duplicate groups to report (default: all).",
    )
    parser.add_argument(
        "--verify-trash", action="store_true",
        help=(
            "Verify that every media file in __duplicate_files_trash/ has a perceptual match in "
            "<directory> within --threshold. Reports any files with no match, "
            "which may indicate idem wrongly trashed them. "
            "Exits with code 1 if unmatched files are found."
        ),
    )
    args = parser.parse_args()

    try:
        delta = parse_size(args.delta)
    except (ValueError, TypeError):
        print(f"Error: invalid --delta value '{args.delta}'.", file=sys.stderr)
        sys.exit(1)

    if args.threshold < 0:
        print("Error: --threshold must be >= 0.", file=sys.stderr)
        sys.exit(1)

    if not (1 <= args.page_size <= 500):
        print(f"Error: --page-size must be between 1 and 500.", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None and args.limit < 1:
        print(f"Error: --limit must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    cache_dir = None
    if args.cache:
        cache_dir = os.path.abspath(args.cache)
        if os.path.exists(cache_dir) and not os.path.isdir(cache_dir):
            print(f"Error: --cache must be a directory, not a file: '{cache_dir}'.",
                  file=sys.stderr)
            sys.exit(1)

    if args.verify_trash:
        verify_trash(directory, args.threshold, cache_dir)
        return

    if args.exact:
        # ── Exact match mode ───────────────────────────────────────────────────
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, DB_FILENAME)
        else:
            db_path = os.path.join(_ensure_db_dir(directory), DB_FILENAME)

        # 1. Scan all media files (images + videos)
        print(f"Scanning {directory} ...")
        all_files = scan_files(directory, MEDIA_EXTENSIONS)
        print(f"  {len(all_files)} media files found")

        # 2. Compute / reuse checksums via all_media_sha_hash_db.csv
        print(f"\nIndexing ...")
        db, new_count, gone_count, errors = build_exact_index(all_files, db_path)
        hits = len(db) - new_count
        print(f"  {len(db)} indexed  *  {new_count} new  *  {hits} hits  "
              f"*  {errors} errors  *  {gone_count} pruned")
        print(f"  DB: {db_path}")

        # 3. Group by checksum
        print(f"\nGrouping by checksum ...")
        abs_path_map = {path_without_drive(p): p for p in all_files}
        groups = group_exact_duplicates(db, abs_path_map)
        print(f"  {len(groups)} duplicate group(s) found")

    else:
        # ── Perceptual hash mode ───────────────────────────────────────────────
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, CACHE_FILENAME)
        else:
            cache_path = os.path.join(_ensure_db_dir(directory), CACHE_FILENAME)

        # 1. Scan
        print(f"Scanning {directory} ...")
        all_files = scan_files(directory)
        print(f"  {len(all_files)} images found")

        # 2. Hash (cache-aware)
        print(f"\nHashing ...")
        cache = load_cache(cache_path)
        live_keys = {path_without_drive(p) for p in all_files}
        cache_before = len(cache)
        cache = {k: v for k, v in cache.items() if k in live_keys}
        gone_count = cache_before - len(cache)

        cache_out = open_cache_for_append(cache_path)
        try:
            hashes, new_count, rehashed_count, errors = build_hashes(all_files, cache, cache_out)
        finally:
            cache_out.close()
        save_cache(cache_path, cache)  # compact: rewrite clean after appending

        hits = len(hashes) - new_count - rehashed_count
        print(f"  {len(hashes)} hashed  *  {new_count} new  *  {rehashed_count} re-hashed  "
              f"*  {hits} hits  *  {errors} errors  *  {gone_count} pruned")
        print(f"  Cache: {cache_path}")

        # 3. Find duplicates
        print(f"\nGrouping (threshold={args.threshold}) ...")
        groups = group_duplicates(hashes, args.threshold)
        print(f"  {len(groups)} duplicate group(s) found")

        # ── Video groups (--video) ─────────────────────────────────────────────
        if args.video:
            if not ffmpeg_available():
                print("Warning: --video requires ffmpeg on PATH; video scan skipped.",
                      file=sys.stderr)
            else:
                if cache_dir:
                    vcache_path = os.path.join(cache_dir, VCACHE_FILENAME)
                else:
                    vcache_path = os.path.join(_ensure_db_dir(directory), VCACHE_FILENAME)

                print(f"\nScanning {directory} for videos ...")
                vid_files = scan_files(directory, VIDEO_EXTENSIONS)
                print(f"  {len(vid_files)} video files found")

                print(f"\nHashing video frames ...")
                vcache = load_vcache(vcache_path)
                live_vkeys = {path_without_drive(p) for p in vid_files}
                vcache = {k: v for k, v in vcache.items() if k in live_vkeys}
                vcache_out = _open_vcache_for_append(vcache_path)
                try:
                    vhashes, vnew, vrehashed, verrors = build_video_hashes(
                        vid_files, vcache, vcache_out)
                finally:
                    vcache_out.close()
                save_vcache(vcache_path, vcache)

                vhits = len(vhashes) - vnew - vrehashed
                print(f"  {len(vhashes)} hashed  *  {vnew} new  *  "
                      f"{vrehashed} re-hashed  *  {vhits} hits  *  {verrors} errors")
                print(f"  Video cache: {vcache_path}")

                print(f"\nGrouping videos (threshold={args.threshold}) ...")
                vid_groups = group_video_duplicates(vhashes, args.threshold)
                print(f"  {len(vid_groups)} duplicate group(s) found")
                groups = groups + vid_groups

    # ── Common: delta filter, limit, output ───────────────────────────────────

    # Apply delta filter (groups are sorted largest-first, so g[0][1] is max).
    if delta > 0:
        before = len(groups)
        groups = [g for g in groups if g[0][1] - g[-1][1] >= delta]
        dropped = before - len(groups)
        if dropped:
            print(f"  {dropped} group(s) dropped (size spread < {fmt_size(delta).strip()})")

    # Apply group limit
    if args.limit is not None and len(groups) > args.limit:
        print(f"  Limiting to first {args.limit} of {len(groups)} group(s) (--limit).")
        groups = groups[:args.limit]

    # Report or review
    if args.review:
        launch_review_ui(groups, directory, page_size=args.page_size, ignore=args.ignore)
    elif args.interactive:
        interactive_mode(groups, directory, ignore=tuple(args.ignore))
    else:
        print_results(groups, directory, exact=args.exact, ignore=tuple(args.ignore))


if __name__ == "__main__":
    main()

# Video perceptual dedup — implementation plan

## Problem

`idem.py` currently has two modes:

| Mode | Files | Matches |
|---|---|---|
| Perceptual (default) | Images only | Near-duplicates via pHash+dHash+BK-tree |
| `--exact` | Images + Videos | Byte-for-byte only |

Videos at different resolutions or bitrates are not matched by either mode.

## Chosen approach: frame-sampling + existing imagehash

Extract N evenly-spaced frames from each video using `ffmpeg`, compute `pHash`
on each frame using the existing Pillow+imagehash stack, and compare videos by
the **mean per-frame Hamming distance** across matched frame positions.

Rationale over alternatives:
- Reuses all existing hashing, caching, and review infrastructure
- No new Python dependencies — only `ffmpeg` on PATH (already expected for video work; hachoir uses it too)
- O(n²) comparison within duration buckets is fine for typical video library sizes (hundreds, not tens of thousands)

---

## New CLI flag

```
--video     Perceptual duplicate detection for video files.
            Uses ffmpeg to sample frames; requires ffmpeg on PATH.
            Can be combined with the existing image scan (default behaviour
            already scans images; --video adds video groups to the output).
            Ignores --exact. Shares --threshold and --cache flags.
```

Invocation examples:
```
python idem.py ~/photos --video
python idem.py ~/photos --video --threshold 10
python idem.py ~/photos --video --review
```

---

## Data model

### Video cache file

A separate CSV alongside the existing `images_perceptual_hash_db.csv` (phash cache):

```
VCACHE_FILENAME = "videos_perceptual_hash_db.csv"
VCACHE_FIELDS   = ["path", "size", "mtime", "duration", "vhash"]
```

- `duration` — float, seconds (from ffprobe/ffmpeg)
- `vhash` — comma-separated hex pHash strings, one per sampled frame
  e.g. `"a1b2c3d4e5f60718,dead0000beef1234,..."`

Cache validity uses the same `size` + `mtime` within `_TS_TOLERANCE` logic as
the image cache. A stale entry is deleted and recomputed.

### N_FRAMES constant

```python
N_VIDEO_FRAMES = 8   # evenly spaced across the video's duration
```

8 frames balances sensitivity against false positives and keeps per-video
hashing fast (< 1 s on most videos with a warm disk cache).

---

## New functions

### `ffmpeg_available() -> bool`

```python
def ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    import shutil
    return shutil.which("ffmpeg") is not None
```

Called once at startup with `--video`; prints an error and exits if missing.

---

### `compute_video_hashes(path, n=N_VIDEO_FRAMES) -> tuple[float, list[int]]`

```python
def compute_video_hashes(path: str, n: int = N_VIDEO_FRAMES) -> tuple:
    """Return (duration_seconds, [phash_int, ...]) for a video file.

    Extracts n frames evenly spaced across the video using ffmpeg.
    Raises RuntimeError if ffmpeg fails or the video cannot be decoded.
    """
```

Implementation sketch:
1. Run `ffprobe -v error -show_entries format=duration -of csv=p=0 <path>`
   to get duration. Fall back to a short ffmpeg decode if ffprobe is absent.
2. Compute timestamps: `t_i = duration * (2*i + 1) / (2*n)` for i in 0..n-1
   (evenly spaced, not at exact start/end to avoid black frames).
3. For each timestamp, run:
   `ffmpeg -ss <t> -i <path> -frames:v 1 -f image2 -vcodec png pipe:1`
   and read the PNG bytes from stdout into a `io.BytesIO`.
4. Open with `Image.open(buf)`, compute `str(imagehash.phash(img))`, collect.
5. Return `(duration, [int(h, 16) for h in phash_strs])`.

Optimisation: all n `ffmpeg` calls can be batched as one call using multiple
`-ss`/`-frames:v 1` segments, or via a single pass with a `select` filter.
Start with the simple n-call version; optimise if it proves slow.

---

### `load_vcache(path) -> dict` / `save_vcache(path, cache) -> None`

Same structure as `load_cache` / `save_cache` for the image phash cache.
`load_vcache` parses `vhash` into `list[str]` (keep as hex strings; convert
to `list[int]` only when needed for comparison).

---

### `build_video_hashes(files, cache, cache_out=None) -> tuple`

```python
def build_video_hashes(files: list, cache: dict, cache_out=None) -> tuple:
    """Compute/load frame hashes for video files.

    Returns (vhashes dict, new_count, rehashed_count, error_count).
    vhashes maps path -> (duration, [phash_int, ...], size_bytes).
    """
```

Structure mirrors `build_hashes` exactly: stat → cache lookup → compute if
stale → write to cache. Shows same progress bar.

---

### `_video_distance(frames_a, frames_b) -> float`

```python
def _video_distance(frames_a: list, frames_b: list) -> float:
    """Mean per-frame pHash Hamming distance between two frame sequences."""
    n = min(len(frames_a), len(frames_b))
    return sum((frames_a[i] ^ frames_b[i]).bit_count() for i in range(n)) / n
```

---

### `group_video_duplicates(vhashes, threshold) -> list`

```python
def group_video_duplicates(vhashes: dict, threshold: int) -> list:
    """Group videos by perceptual similarity.

    Pre-filters by duration (within 5%) before doing full frame comparison.
    Returns same format as group_duplicates: list of [(path, size), ...].
    """
```

Algorithm:
1. Sort videos by duration.
2. For each video A not yet assigned to a group, collect candidates B where
   `abs(dur_A - dur_B) / max(dur_A, dur_B) <= 0.05` (5% duration window).
3. Within that window, compute `_video_distance(frames_A, frames_B)`.
4. If distance <= threshold, they are duplicates.
5. Build connected components (union-find or DFS) over the similarity edges.
6. Return groups sorted largest-first by file size, groups sorted
   most-files-first — same ordering as `group_duplicates`.

O(n²) within each duration bucket is acceptable; video libraries are typically
small. Note: BK-tree is not used here because the distance metric operates on
a *sequence* of hashes, not a single integer.

---

## Integration points

### `main()` — CLI path for `--video`

```python
if args.video:
    if not ffmpeg_available():
        print("Error: ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {directory} for videos ...")
    vid_files = scan_files(directory, VIDEO_EXTENSIONS)
    print(f"  {len(vid_files)} video files found")

    vcache_path = os.path.join(directory, VCACHE_FILENAME)
    vcache = load_vcache(vcache_path)

    print(f"\nHashing video frames ...")
    vcache_out = open(vcache_path, "a", newline="", encoding="utf-8")
    vhashes, new_count, rehashed, errors = build_video_hashes(vid_files, vcache, vcache_out)
    vcache_out.close()
    save_vcache(vcache_path, vcache)
    print(f"  {len(vhashes)} hashed  *  {new_count} new  *  {errors} errors")

    print(f"\nGrouping by visual similarity (threshold={args.threshold}) ...")
    vid_groups = group_video_duplicates(vhashes, args.threshold)
    print(f"  {len(vid_groups)} duplicate group(s) found")

    # If also running image scan (default), merge and display together,
    # or display video groups standalone if --video only.
```

### `print_results` / review UI

No changes needed:
- `print_results` already accepts any groups list of `[(path, size), ...]`.
- The review UI already sets `"is_video": True` in `_groups_to_json` for
  files with video extensions, and shows a placeholder instead of a thumbnail.
  Video groups will display the same way as exact-mode video groups today.

### `verify_trash`

Extend the video verification branch: after the exact-checksum check, also
run a perceptual check for any trash video with no exact match. This covers
the case where a re-encoded copy was kept and the original trashed.

---

## Cache schema summary

| File | Fields | Used by |
|---|---|---|
| `images_perceptual_hash_db.csv` | path, size, mtime, phash, dhash | Image perceptual mode (existing) |
| `videos_perceptual_hash_db.csv` | path, size, mtime, duration, vhash | Video perceptual mode (new) |
| `kura_db.csv` | path, size, ctime, mtime, checksum | Exact mode, shared with kura.py |

---

## Threshold guide (video mode)

Same Hamming scale as image mode, but now the mean across N frames:

| Threshold | Meaning |
|---|---|
| 0 | Exact same visual frames (different container/codec only) |
| 5 | Same video, minor re-encode or slight colour grade change |
| 10 | Same video, noticeably different resolution or bitrate (default) |
| 20 | Loose — covers significant colour/crop differences; risk of false positives |

---

## Dependencies

| Dependency | Status | Notes |
|---|---|---|
| Pillow | Already required | Used for frame hashing |
| imagehash | Already required | pHash computation |
| pybktree | Already required | Not used for video (O(n²) instead) |
| ffmpeg | New, external | Must be on PATH; not a pip package |

No new `pip install` requirements.

---

## Implementation order

1. `ffmpeg_available()` + `compute_video_hashes()` + manual test on a few files
2. `load_vcache()` / `save_vcache()` + round-trip test
3. `build_video_hashes()` with progress bar + cache persistence
4. `_video_distance()` + `group_video_duplicates()`
5. Wire into `main()` behind `--video` flag
6. Extend `verify_trash()` to use perceptual matching for videos
7. Update docstring and `--help` text

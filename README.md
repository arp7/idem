# idem.py — Find Duplicate Images

### Why would I want to use idem?
You are a photographer with a media collection that has duplicate images/videos and you want to retain certain copies e.g. the ones with the highest resolution.

### What are duplicate images or vides?
The same photo or video at either the same or different resolutions, compression levels, or formats.

### How does idem help me?
`idem` determines which images and videos "look" the same and groups all copies identical media together. It then optionally presents these groups together in your default internet browser and for each group it lets you select one or more copies to retain. It suggests retaining only the highest resolution replica by default, and it also attempts to pick the best folder and media file name from the options by applying some simple heuristics. You are free to override all or none of these choices.

By default `idem` just detects and prints out information about groups of "matching" media files. It supports two modes to delete duplicates - `--review` which allows you to visually see the files in your browser, and `--interactive` which prompts you for each group in the terminal. Most users will want to use the `--review` mode.

### What are the limitations?
`idem` uses two different algorithms called `pHash` (Perceptual Hashing) and `dHash` (Difference Hashing). These algorithms are not perfect, and images that are very slightly different may be detected as duplicates. E.g. photos of a slowly moving subject shot in burst mode.

The tool has an `exact` match mode where it checks that the replicas are exactly the same. This will not detect copies of the same image or video at different resolutions. 

### Any other caveats?
The tool assumes you are comfortable installing the [Python Language](https://www.python.org/downloads/) on your machine, installing `idem` dependencies and invoking the tool from the command-line.

### How does idem ensure the safety of my data?
Removed files are moved to a sub-directory named `__duplicate_files_trash/` — they are never deleted outright. So you can always recover them manually or choose to permanently delete the trashed files yourself.

`idem` also enforces that at least one replica of each image must be retained.

`idem` runs locally on your computer and it will never transmit any data over the internet to anyone, ever.  

## Modes

| Mode | Flag | What it detects |
|------|------|----------------|
| **Perceptual** (default) | *(none)* | Visually identical images using pHash + dHash |
| **Video** | `--video` | Visually similar videos by sampling 8 frames with ffmpeg |
| **Exact** | `--exact` | Byte-for-byte identical media files (images + videos) via SHA-256 |

`--video` adds video groups on top of the default image scan. `--exact` replaces the perceptual scan entirely and covers all media.

## Supported Formats

**Images (perceptual and exact modes):** JPEG, PNG, GIF, BMP, TIFF, WebP, HEIC/HEIF.

**Videos (exact and `--video` modes):** MP4, MOV, AVI, MKV, WMV, WebM, FLV, 3GP, M4V, MTS/M2TS.

**Not supported:**
- RAW camera files (`.cr2`, `.nef`, `.arw`, `.dng`, etc.) — most photographers will not want to eliminate original RAW files, so they are explicitly ignored.

## Requirements

- Python 3.10+
- [Pillow](https://python-pillow.org/) (required)
- [imagehash](https://github.com/JohannesBuchner/imagehash) (required)
- [pybktree](https://github.com/benhoyt/pybktree) (required)
- [Flask](https://flask.palletsprojects.com/) (optional — required for `--review`)
- [ffmpeg](https://ffmpeg.org/download.html) (optional — required for `--video`)

```bash
pip install Pillow imagehash pybktree
pip install flask   # optional, for --review
```

## Usage

```bash
python idem.py <directory> [options]
```

### Core options

| Argument | Description |
|----------|-------------|
| `directory` | Directory with your photos and images. It will be scanned recursively |
| `--exact` | SHA-256 exact-match mode — covers all media including videos. Ignores `--threshold` |
| `--video` | Add perceptual video scan on top of the image scan (requires ffmpeg on PATH) |
| `--limit N` | Maximum number of duplicate groups to report (default: all) |

### Output options

| Argument | Description |
|----------|-------------|
| `--review` | Launch a local browser UI to review duplicates and move unwanted files to `__duplicate_files_trash/`. Requires Flask |
| `--interactive` | Step through each group in the terminal. Auto-selects the best filename; you pick which directory to keep (a/b/c…) or press `s` to skip |
| `--page-size N` | Groups per page in `--review` mode (1–500, default: 10) |
| `--ignore WORD` | Treat WORD as noise when auto-scoring filenames and folder names, so it never influences which copy is pre-selected. Case-insensitive. Repeatable: `--ignore backup --ignore resized` |

### Diagnostic options

| Argument | Description |
|----------|-------------|
| `--verify-trash` | Check that every file in `__duplicate_files_trash/` has a perceptual match in `<directory>`. Reports files with no match (potential incorrect trashing). Exits with code 1 if any are found |

### Advanced options
Most users will not need these options and changing them is not recommended. The most interesting of these options is `--threshold`. Increeasing the threshold will allow _less similar_ images to be detected as duplicates and could be useful to find visually similar images in your collection. However it should be used with extreme care.

| Argument | Description |
|----------|-------------|
| `--threshold N` | Hamming distance threshold for perceptual mode (default: `0` — see [Threshold Guide](#threshold-guide)). _If you are unsure then just leave it at the default value_. |
| `--delta SIZE` | Only report groups where largest − smallest ≥ SIZE. Accepts `kb`/`mb`/`gb` suffix (e.g. `100`, `50kb`, `2mb`). Default: `0`. Has no effect in `--exact` mode (exact duplicates are always the same size) |
| `--cache DIR` | Directory in which to store hash database files (default: `<directory>/__databases/`). Applies to all modes including `--exact` |

## Example Usages

**Scan a photo library:**

```bash
python idem.py /mnt/external/Photos
```

**Only report groups with a meaningful size difference:**

```bash
python idem.py /mnt/external/Photos --delta 2mb
```

**Add perceptual video deduplication:**

```bash
python idem.py /mnt/external/Photos --video
```

**Exact-match mode (all media, byte-for-byte):**

```bash
python idem.py /mnt/external/Photos --exact
```

**Browser review UI — 20 groups per page, skip small size differences:**

```bash
python idem.py /mnt/external/Photos --review --page-size 20 --delta 500kb
```

**Review only the first 50 groups, ignoring noisy folder names:**

```bash
python idem.py /mnt/external/Photos --review --limit 50 --ignore backup --ignore resized
```

**Step through groups interactively in the terminal:**

```bash
python idem.py /mnt/external/Photos --interactive
```

**Store the cache in a custom location:**

```bash
python idem.py /mnt/external/Photos --cache /tmp/my_cache_dir/
```

## Threshold Guide

The threshold is the maximum Hamming distance between two hashes for them to be considered duplicates. Applies to perceptual image mode and `--video` mode.

| Threshold | What it catches |
|-----------|----------------|
| `0` (default) | Exact visual duplicates — same image at different resolutions or formats |
| `5` | Slightly edited versions (e.g. minor JPEG re-saves) |
| `10` | Visually similar images (e.g. successive burst shots with small motion) |
| `>10` | High risk of false positives |

Using a non-zero threshold is not recommended for unattended runs — even threshold 0 can produce false positives (e.g. subject's eyes open vs. closed in successive burst shots). Non-zero thresholds also slow down duplicate detection significantly.

For `--video`, the threshold is the mean per-frame Hamming distance:

| Threshold | What it catches |
|-----------|----------------|
| `0` | Same video in a different container or codec |
| `5` | Same video with a minor re-encode or colour grade |
| `10` | Same video at a different resolution or bitrate |

## Review UI Decision Logic

The `--review` and `--interactive` modes apply heuristics to auto-select which copy to keep:

- **Resolution**: the highest-resolution (largest) copy is kept by default.
- **Filename score**: names with English words score higher than pure numbers. Camera-generated prefixes (IMG, DCIM, PXL, DSC, …) score zero. Pass `--ignore WORD` to add custom noise patterns.
- **Folder score**: folder names with meaningful words score higher than generic names.

**Example:** for three visually identical images:

| Image Path | Size |
|------------|------|
| `/photos/IMG_20210705.jpg` | 7.8 MB |
| `/photos/vacation/IMG_20210705.jpg` | 592 KB |
| `/photos/new_york_5.jpg` | 592 KB |

The UI will keep the 7.8 MB copy, rename it `new_york_5.jpg`, and move it to `vacation/`. Default actions:
1. Delete `/photos/new_york_5.jpg` (low-resolution replica)
2. Delete `/photos/vacation/IMG_20210705.jpg` (low-resolution replica)
3. Move `/photos/IMG_20210705.jpg` → `/photos/vacation/new_york_5.jpg`

All choices can be overridden in the review UI.

## Output

```
Found 2 duplicate group(s)  ·  5 files  ·  12.3 MB potentially recoverable

────────────────────────────────────────────────────────────────────────────────

Group 1  ·  3 files
   3.2 MB  vacation/beach.jpg  ← largest
           /Photos/vacation/beach.jpg
   1.8 MB  backup/beach_compressed.jpg
           /Photos/backup/beach_compressed.jpg
   0.3 MB  thumbs/beach_sm.jpg
           /Photos/thumbs/beach_sm.jpg

Group 2  ·  2 files
   4.1 MB  family/birthday.jpg  ← largest
           /Photos/family/birthday.jpg
   0.8 MB  social/birthday_web.png
           /Photos/social/birthday_web.png
```

Files are sorted largest-first within each group. The largest is most likely the original. The summary line shows how much space could be freed by removing the smaller copies.

## How It Works

### Perceptual mode (default)

1. **Scan** the directory recursively for supported image files (skipping `__duplicate_files_trash/` and `__databases/`).
2. **Load** the hash cache (`__databases/images_perceptual_hash_db.csv`).
3. **Hash** each file using both pHash (DCT-based perceptual hash) and dHash (gradient-based difference hash). Files whose size and mtime match the cache are not re-hashed. Each new hash is written to the cache immediately and flushed to disk every 200 entries.
4. **Compact** the cache to a clean single-entry-per-file CSV.
5. **Build a BK-tree** over all pHashes and query it to find all pairs within the threshold — O(n log n). Each candidate pair is confirmed with a secondary dHash check.
6. **Report** each group of near-duplicate files, optionally filtering by `--delta`.

### Video mode (`--video`)

Samples 8 evenly-spaced frames per video using ffmpeg, computes a pHash per frame, and groups videos whose mean per-frame Hamming distance is within `--threshold`. Frame hashes are cached in `__databases/videos_perceptual_hash_db.csv`. Videos differing in duration by more than `max(10 s, 5%)` are never compared.

### Exact mode (`--exact`)

Computes SHA-256 checksums for all media files (images + videos). Files larger than 12 MiB are sampled from three 4 MiB windows (start, middle, end) for speed while remaining compatible with an existing shared checksum database. Checksums are stored in `__databases/all_media_sha_hash_db.csv`.

## Hash Caches

All three caches live in `__databases/` inside the scanned directory:

| File | Mode | Contents |
|------|------|---------|
| `images_perceptual_hash_db.csv` | Perceptual | path, size, mtime → pHash + dHash |
| `videos_perceptual_hash_db.csv` | `--video` | path, size, mtime → per-frame hashes |
| `all_media_sha_hash_db.csv` | `--exact` | path, size, mtime → SHA-256 checksum |

On each run, files whose size and mtime are unchanged are served from the cache — no image I/O needed. Because hashes are flushed incrementally, the cache remains valid even if a run is interrupted. The first run for a large collection (~100 000 images) may take 2–3 hours; subsequent runs only re-hash new or changed files.

## Running Tests

```bash
pip install pytest
python -m pytest test_idem.py -q
```

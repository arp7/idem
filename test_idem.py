"""Tests for idem.py"""

import csv
import io
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
import imagehash

import idem as idem_module
from idem import (
    CACHE_FIELDS,
    CACHE_FILENAME,
    DB_DIR,
    N_VIDEO_FRAMES,
    VCACHE_FIELDS,
    VCACHE_FILENAME,
    _FULL_HASH_THRESHOLD,
    _TS_TOLERANCE,
    _collect_stale,
    _file_checksum,
    _folder_score,
    _get_video_duration,
    _name_score,
    _open_vcache_for_append,
    _resolve_transform,
    _smart_defaults,
    _valid_hex,
    _video_distance,
    build_exact_index,
    build_hashes,
    build_video_hashes,
    compute_video_hashes,
    ffmpeg_available,
    fmt_size,
    group_duplicates,
    group_exact_duplicates,
    group_video_duplicates,
    load_cache,
    load_vcache,
    open_cache_for_append,
    parse_size,
    path_without_drive,
    save_cache,
    save_vcache,
    scan_files,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_image(color="red", size=(32, 32), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def write_image(path: Path, color="red", size=(32, 32)) -> Path:
    path.write_bytes(make_image(color=color, size=size))
    return path


def fake_hash(hex_str: str) -> imagehash.ImageHash:
    """Create an ImageHash directly from a hex string (no image I/O needed)."""
    return imagehash.hex_to_hash(hex_str)


def fake_entry(phash_hex: str, size: int = 100) -> tuple:
    """Return a (phash_int, dhash_int, size) tuple as produced by build_hashes.

    dhash is set to 0 so the secondary filter in group_duplicates always passes,
    keeping unit tests focused on phash-based grouping logic.
    """
    return (int(phash_hex, 16), 0, size)


# ── path_without_drive ─────────────────────────────────────────────────────────

class TestPathWithoutDrive:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific")
    def test_strips_drive_letter(self):
        """The 'C:' prefix is removed, leaving the rest of the path intact."""
        assert path_without_drive("C:\\Users\\foo\\bar.jpg") == "\\Users\\foo\\bar.jpg"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific")
    def test_strips_various_drive_letters(self):
        """Drive-letter stripping works for any letter (D:, Z:, etc.), not just C:."""
        assert path_without_drive("D:\\photos\\beach.jpg") == "\\photos\\beach.jpg"
        assert path_without_drive("Z:\\archive\\img.png") == "\\archive\\img.png"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific")
    def test_already_driveless_unchanged(self):
        """A path that already has no drive letter is returned unchanged."""
        p = "\\Users\\foo\\bar.jpg"
        assert path_without_drive(p) == p

    @pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows specific")
    def test_unix_path_unchanged(self):
        """Unix-style paths have no drive letter and are returned as-is."""
        p = "/Users/foo/bar.jpg"
        assert path_without_drive(p) == p

    def test_idempotent(self, tmp_path):
        """Applying twice gives the same result as applying once."""
        p = str(tmp_path / "photo.jpg")
        assert path_without_drive(path_without_drive(p)) == path_without_drive(p)

    def test_result_has_no_drive_letter(self, tmp_path):
        """Works on both platforms — result never has X: prefix."""
        result = path_without_drive(str(tmp_path / "photo.jpg"))
        assert not (len(result) >= 2 and result[1] == ":")


# ── fmt_size ───────────────────────────────────────────────────────────────────

class TestFmtSize:
    def test_bytes(self):
        """Values below 1 KB are formatted with a 'B' suffix."""
        assert "B" in fmt_size(512)

    def test_kilobytes(self):
        """Values in the KB range are formatted with a 'KB' suffix."""
        assert "KB" in fmt_size(2 * 1024)

    def test_megabytes(self):
        """Values in the MB range are formatted with a 'MB' suffix."""
        assert "MB" in fmt_size(5 * 1024 * 1024)

    def test_gigabytes(self):
        """Values in the GB range are formatted with a 'GB' suffix."""
        assert "GB" in fmt_size(3 * 1024 ** 3)


# ── load_cache / save_cache ────────────────────────────────────────────────────

class TestCacheIO:
    def test_missing_file_returns_empty(self, tmp_path):
        """A non-existent cache file returns an empty dict rather than raising."""
        assert load_cache(str(tmp_path / "nonexistent.csv")) == {}

    def test_round_trip_preserves_data(self, tmp_path):
        """Data saved with save_cache and reloaded with load_cache is identical."""
        cache_path = str(tmp_path / "cache.csv")
        original = {
            "\\photos\\beach.jpg": {"size": 1000, "mtime": 1708531200.0, "phash": "f8c8e4e2c4e4e8f0", "dhash": ""},
            "\\photos\\other.png": {"size": 2000, "mtime": 1708531300.5, "phash": "a0b0c0d0e0f0a0b0", "dhash": ""},
        }
        save_cache(cache_path, original)
        loaded = load_cache(cache_path)
        assert loaded == original

    def test_load_normalizes_types(self, tmp_path):
        """size must be int, mtime must be float after loading."""
        cache_path = str(tmp_path / "cache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
            writer.writeheader()
            writer.writerow({"path": "\\photos\\a.jpg", "size": "1234",
                             "mtime": "1708531200.5", "phash": "0000000000000000"})
        entry = load_cache(cache_path)["\\photos\\a.jpg"]
        assert isinstance(entry["size"], int)
        assert isinstance(entry["mtime"], float)

    @pytest.mark.skipif(sys.platform != "win32", reason="Drive letters only on Windows")
    def test_save_writes_keys_as_is(self, tmp_path):
        """save_cache writes keys verbatim; callers normalise via path_without_drive."""
        cache_path = str(tmp_path / "cache.csv")
        # Keys are always drive-stripped before insertion (see build_hashes /
        # build_exact_index), so save_cache receives a driveless key here.
        cache = {"\\photos\\beach.jpg": {"size": 1000, "mtime": 1.0, "phash": "0" * 16}}
        save_cache(cache_path, cache)

        with open(cache_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["path"] == "\\photos\\beach.jpg"

    @pytest.mark.skipif(sys.platform != "win32", reason="Drive letters only on Windows")
    def test_load_strips_drive_letter_from_csv(self, tmp_path):
        """Drive letters already in the CSV are stripped on read (migration safety)."""
        cache_path = str(tmp_path / "cache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
            writer.writeheader()
            writer.writerow({"path": "C:\\photos\\beach.jpg", "size": 1000,
                             "mtime": 1.0, "phash": "0" * 16})
        cache = load_cache(cache_path)
        assert "C:\\photos\\beach.jpg" not in cache
        key = next(iter(cache))
        assert not key.startswith("C:")
        assert "beach.jpg" in key

    def test_corrupt_file_returns_empty(self, tmp_path, capsys):
        """A file with invalid CSV content returns an empty dict and does not raise."""
        cache_path = str(tmp_path / "cache.csv")
        Path(cache_path).write_text("not_a_field\ngarbage_row", encoding="utf-8")
        # Should not raise; returns empty dict and prints a warning
        cache = load_cache(cache_path)
        assert isinstance(cache, dict)


# ── scan_files ─────────────────────────────────────────────────────────────────

class TestScanFiles:
    def test_finds_common_image_extensions(self, tmp_path):
        """All common raster image formats (.jpg, .png, .gif, etc.) are discovered."""
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]:
            (tmp_path / f"photo{ext}").write_bytes(b"x")
        found = {Path(f).suffix.lower() for f in scan_files(str(tmp_path))}
        assert found == {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

    def test_finds_heic_heif(self, tmp_path):
        """HEIC and HEIF (Apple mobile) formats are also discovered."""
        (tmp_path / "photo.heic").write_bytes(b"x")
        (tmp_path / "photo.heif").write_bytes(b"x")
        assert len(scan_files(str(tmp_path))) == 2

    def test_ignores_raw_extensions(self, tmp_path):
        """Raw camera formats (.cr2, .nef, etc.) are excluded from scan results."""
        for ext in [".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2"]:
            (tmp_path / f"raw{ext}").write_bytes(b"x")
        assert scan_files(str(tmp_path)) == []

    def test_ignores_video_extensions(self, tmp_path):
        """Video files (.mp4, .mov, etc.) are excluded from image scan results."""
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".m4v"]:
            (tmp_path / f"video{ext}").write_bytes(b"x")
        assert scan_files(str(tmp_path)) == []

    def test_ignores_non_media_files(self, tmp_path):
        """Non-media files (.txt, .csv) are excluded from scan results."""
        (tmp_path / "readme.txt").write_bytes(b"x")
        (tmp_path / "data.csv").write_bytes(b"x")
        assert scan_files(str(tmp_path)) == []

    def test_recursive_discovery(self, tmp_path):
        """Files in nested subdirectories are found alongside top-level files."""
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (tmp_path / "top.jpg").write_bytes(b"x")
        (sub / "deep.jpg").write_bytes(b"x")
        assert len(scan_files(str(tmp_path))) == 2

    def test_case_insensitive_extension(self, tmp_path):
        """Extension matching is case-insensitive (.JPG is found just like .jpg)."""
        (tmp_path / "photo.JPG").write_bytes(b"x")
        (tmp_path / "photo.PNG").write_bytes(b"x")
        assert len(scan_files(str(tmp_path))) == 2

    def test_returns_sorted(self, tmp_path):
        """The returned file list is in alphabetical order."""
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            (tmp_path / name).write_bytes(b"x")
        found = scan_files(str(tmp_path))
        assert found == sorted(found)

    def test_returns_absolute_paths(self, tmp_path):
        """Every returned path is an absolute filesystem path."""
        (tmp_path / "photo.jpg").write_bytes(b"x")
        found = scan_files(str(tmp_path))
        assert all(os.path.isabs(p) for p in found)


# ── build_hashes ───────────────────────────────────────────────────────────────

class TestBuildHashes:
    def test_computes_hash_for_new_file(self, tmp_path):
        """A file not in the cache is hashed and counted as new."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        hashes, new_c, rehashed_c, errors = build_hashes([str(p)], cache)
        assert len(hashes) == 1
        assert new_c == 1 and rehashed_c == 0
        assert errors == 0

    def test_cache_key_is_driveless(self, tmp_path):
        """Cache keys are stored without drive letters for cross-platform consistency."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        build_hashes([str(p)], cache)
        for key in cache:
            assert not (len(key) >= 2 and key[1] == ":"), f"Drive letter found in key: {key!r}"

    def test_cache_hit_skips_recompute(self, tmp_path):
        """A file with unchanged size/mtime is served from cache with no recompute."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        hashes1, updated1, _, _ = build_hashes([str(p)], cache)
        assert updated1 == 1

        hashes2, updated2, _, _ = build_hashes([str(p)], cache)
        assert updated2 == 0
        assert hashes1[str(p)] == hashes2[str(p)]

    def test_stale_size_triggers_recompute(self, tmp_path):
        """A file that has grown in size is recomputed and counted as re-hashed."""
        p = write_image(tmp_path / "photo.jpg", size=(32, 32))
        cache = {}
        build_hashes([str(p)], cache)

        # Overwrite with a larger image (different size)
        write_image(p, color="blue", size=(64, 64))
        _, _, updated, _ = build_hashes([str(p)], cache)
        assert updated == 1

    def test_stale_mtime_triggers_recompute(self, tmp_path):
        """A file whose cached mtime differs beyond tolerance is recomputed."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        build_hashes([str(p)], cache)

        # Manually corrupt the cached mtime to force staleness
        key = path_without_drive(str(p))
        cache[key]["mtime"] -= _TS_TOLERANCE + 10

        _, _, updated, _ = build_hashes([str(p)], cache)
        assert updated == 1

    def test_non_image_content_counted_as_error(self, tmp_path):
        """A file that cannot be opened as an image is counted as an error, not re-hashed."""
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"this is not an image")
        cache = {}
        _, updated, _, errors = build_hashes([str(p)], cache)
        assert errors == 1
        assert updated == 0

    def test_multiple_files(self, tmp_path):
        """Multiple files are all hashed in a single call, each counted independently."""
        files = [write_image(tmp_path / f"photo{i}.jpg", color=("red", "green", "blue")[i])
                 for i in range(3)]
        cache = {}
        hashes, updated, _, errors = build_hashes([str(f) for f in files], cache)
        assert len(hashes) == 3
        assert updated == 3
        assert errors == 0

    def test_cache_populated_after_run(self, tmp_path):
        """After hashing, the cache contains size, mtime, and phash for each file."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        build_hashes([str(p)], cache)
        key = path_without_drive(str(p))
        assert key in cache
        assert "phash" in cache[key]
        assert "size" in cache[key]
        assert "mtime" in cache[key]


# ── group_duplicates ───────────────────────────────────────────────────────────

class TestGroupDuplicates:
    """Uses fake ImageHash objects to test grouping logic without image I/O."""

    def _files(self, tmp_path, *names, size=100):
        """Create dummy files (content doesn't matter for grouping logic)."""
        paths = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"x" * size)
            paths.append(str(p))
        return paths

    def test_empty_input(self):
        """An empty hash dict returns no duplicate groups."""
        assert group_duplicates({}, threshold=10) == []

    def test_single_file_not_grouped(self, tmp_path):
        """A single file cannot form a duplicate group."""
        (p,) = self._files(tmp_path, "a.jpg")
        assert group_duplicates({p: fake_entry("0" * 16)}, threshold=10) == []

    def test_identical_hashes_grouped(self, tmp_path):
        """Two files with the same hash are placed in one duplicate group."""
        p1, p2 = self._files(tmp_path, "a.jpg", "b.jpg")
        h = fake_entry("0" * 16)
        groups = group_duplicates({p1: h, p2: h}, threshold=0)
        assert len(groups) == 1
        assert {p for p, _ in groups[0]} == {p1, p2}

    def test_distant_hashes_not_grouped(self, tmp_path):
        """Files with maximum Hamming distance (all bits differ) are not grouped."""
        p1, p2 = self._files(tmp_path, "a.jpg", "b.jpg")
        # Max Hamming distance = 64
        groups = group_duplicates(
            {p1: fake_entry("0" * 16), p2: fake_entry("f" * 16)},
            threshold=10,
        )
        assert groups == []

    def test_near_duplicate_within_threshold(self, tmp_path):
        """Files within the Hamming distance threshold are grouped as near-duplicates."""
        p1, p2 = self._files(tmp_path, "a.jpg", "b.jpg")
        # "000000000000000f" vs "0000000000000000": 4 bits differ → distance = 4
        groups = group_duplicates(
            {p1: fake_entry("000000000000000f"), p2: fake_entry("0000000000000000")},
            threshold=5,
        )
        assert len(groups) == 1

    def test_near_duplicate_outside_threshold_not_grouped(self, tmp_path):
        """Files whose Hamming distance exceeds the threshold are not grouped."""
        p1, p2 = self._files(tmp_path, "a.jpg", "b.jpg")
        # "000000000000000f" vs "0000000000000000": distance = 4
        groups = group_duplicates(
            {p1: fake_entry("000000000000000f"), p2: fake_entry("0000000000000000")},
            threshold=3,
        )
        assert groups == []

    def test_hamming_distance_exact_boundary(self, tmp_path):
        """bit_count() Hamming distance is computed correctly at the threshold boundary."""
        p1, p2 = self._files(tmp_path, "a.jpg", "b.jpg")
        # 0x000000000000000f has exactly 4 set bits → Hamming distance 4 from 0x0
        ph_4bits = 0x000000000000000f
        assert (ph_4bits ^ 0).bit_count() == 4   # verify the bit pattern itself
        hashes = {p1: (ph_4bits, 0, 100), p2: (0, 0, 100)}
        assert group_duplicates(hashes, threshold=4) != []  # 4 <= 4: grouped
        assert group_duplicates(hashes, threshold=3) == []  # 4 >  3: not grouped

    def test_within_group_sorted_largest_first(self, tmp_path):
        """Within a duplicate group, the largest file (by byte size) appears first."""
        large, small = self._files(tmp_path, "large.jpg", "small.jpg")
        h_large = fake_entry("0" * 16, size=1000)
        h_small = fake_entry("0" * 16, size=100)
        groups = group_duplicates({large: h_large, small: h_small}, threshold=0)
        assert len(groups) == 1
        assert groups[0][0][0] == large   # largest first
        assert groups[0][0][1] == 1000
        assert groups[0][1][1] == 100

    def test_multiple_groups_detected(self, tmp_path):
        """Two independent clusters of similar images each produce a separate group."""
        p1, p2 = self._files(tmp_path, "a1.jpg", "a2.jpg")
        p3, p4 = self._files(tmp_path, "b1.jpg", "b2.jpg")
        ha = fake_entry("0" * 16)
        hb = fake_entry("f" * 16)
        groups = group_duplicates({p1: ha, p2: ha, p3: hb, p4: hb}, threshold=0)
        assert len(groups) == 2

    def test_groups_sorted_most_files_first(self, tmp_path):
        """Groups with more files appear before groups with fewer files."""
        trio  = self._files(tmp_path, "a1.jpg", "a2.jpg", "a3.jpg")
        pair  = self._files(tmp_path, "b1.jpg", "b2.jpg")
        ha = fake_entry("0" * 16)
        hb = fake_entry("f" * 16)
        hashes = {p: ha for p in trio} | {p: hb for p in pair}
        groups = group_duplicates(hashes, threshold=0)
        assert len(groups) == 2
        assert len(groups[0]) == 3   # trio comes first
        assert len(groups[1]) == 2

    def test_no_file_appears_in_two_groups(self, tmp_path):
        """Each file is claimed by at most one group."""
        files = self._files(tmp_path, *[f"f{i}.jpg" for i in range(6)])
        # All files share the same hash → one big group
        h = fake_entry("0" * 16)
        hashes = {p: h for p in files}
        groups = group_duplicates(hashes, threshold=0)
        all_paths = [p for g in groups for p, _ in g]
        assert len(all_paths) == len(set(all_paths))


# ── Cache persistence across runs ──────────────────────────────────────────────

class TestCachePersistence:
    def test_second_run_hits_cache(self, tmp_path):
        """On the second call with the same cache, nothing is recomputed."""
        p = write_image(tmp_path / "photo.jpg")
        cache_path = str(tmp_path / CACHE_FIELDS[0])   # reuse tmp_path cleanly
        cache_path = str(tmp_path / "cache.csv")

        cache = {}
        _, updated1, _, _ = build_hashes([str(p)], cache)
        save_cache(cache_path, cache)
        assert updated1 == 1

        cache2 = load_cache(cache_path)
        _, updated2, _, _ = build_hashes([str(p)], cache2)
        assert updated2 == 0

    def test_saved_cache_has_no_drive_letters(self, tmp_path):
        """After save_cache, the CSV path column contains no drive letters."""
        p = write_image(tmp_path / "photo.jpg")
        cache_path = str(tmp_path / "cache.csv")
        cache = {}
        build_hashes([str(p)], cache)
        save_cache(cache_path, cache)

        with open(cache_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert not (len(row["path"]) >= 2 and row["path"][1] == ":"), \
                f"Drive letter found in saved path: {row['path']!r}"

    def test_hash_survives_round_trip(self, tmp_path):
        """Hash loaded from cache matches the originally computed hash."""
        p = write_image(tmp_path / "photo.jpg")
        cache_path = str(tmp_path / "cache.csv")

        cache1 = {}
        hashes1, _, _, _ = build_hashes([str(p)], cache1)
        save_cache(cache_path, cache1)

        cache2 = load_cache(cache_path)
        hashes2, _, _, _ = build_hashes([str(p)], cache2)

        assert hashes1[str(p)] == hashes2[str(p)]

    def test_prune_removes_missing_files(self, tmp_path):
        """Cache entries for files no longer on disk are pruned."""
        keep = write_image(tmp_path / "keep.jpg", color="red")
        drop = write_image(tmp_path / "drop.jpg", color="blue")

        cache = {}
        build_hashes([str(keep), str(drop)], cache)
        assert len(cache) == 2

        # Simulate next run: only 'keep' is present
        all_files = [str(keep)]
        live_keys = {path_without_drive(p) for p in all_files}
        gone = [k for k in cache if k not in live_keys]
        for k in gone:
            del cache[k]

        assert len(cache) == 1
        assert all("keep.jpg" in k for k in cache)

    def test_modified_file_recomputed_after_cache_reload(self, tmp_path):
        """After save/load cycle, a modified file is detected as stale."""
        p = write_image(tmp_path / "photo.jpg", color="red", size=(32, 32))
        cache_path = str(tmp_path / "cache.csv")

        cache = {}
        build_hashes([str(p)], cache)
        save_cache(cache_path, cache)

        # Modify the file between runs
        write_image(p, color="blue", size=(64, 64))

        cache2 = load_cache(cache_path)
        _, _, updated, _ = build_hashes([str(p)], cache2)
        assert updated == 1


# ── End-to-end: duplicate detection on real images ─────────────────────────────

class TestEndToEnd:
    def test_same_image_different_sizes_detected(self, tmp_path):
        """Same visual content at two resolutions → reported as duplicates."""
        large = write_image(tmp_path / "large.jpg", color="red", size=(64, 64))
        small = write_image(tmp_path / "small.jpg", color="red", size=(32, 32))
        cache = {}
        hashes, _, _, _ = build_hashes([str(large), str(small)], cache)
        groups = group_duplicates(hashes, threshold=10)
        assert len(groups) == 1
        paths = {p for p, _ in groups[0]}
        assert str(large) in paths
        assert str(small) in paths

    def test_different_images_not_grouped(self, tmp_path):
        """Visually distinct images should not end up in the same group."""
        # Use a real image vs a checkerboard for maximum DCT difference
        solid = write_image(tmp_path / "solid.jpg", color="white", size=(64, 64))

        checker = Image.new("RGB", (64, 64))
        px = checker.load()
        for y in range(64):
            for x in range(64):
                px[x, y] = (0, 0, 0) if (x // 8 + y // 8) % 2 == 0 else (255, 255, 255)
        checker_path = tmp_path / "checker.jpg"
        checker.save(str(checker_path), format="JPEG")

        cache = {}
        hashes, _, _, _ = build_hashes([str(solid), str(checker_path)], cache)
        ph_solid   = hashes[str(solid)][0]
        ph_checker = hashes[str(checker_path)][0]
        dist = (ph_solid ^ ph_checker).bit_count()
        # Only assert grouping for threshold below actual distance
        if dist > 0:
            groups = group_duplicates(hashes, threshold=dist - 1)
            assert groups == []


# ── Incremental cache writes ────────────────────────────────────────────────────

class TestIncrementalCache:
    """Verify that hashes are flushed to disk as each file is processed."""

    def test_open_cache_for_append_creates_file_with_header(self, tmp_path):
        """Opening a new cache file for append creates it and writes the CSV header row."""
        cache_path = str(tmp_path / "cache.csv")
        f = open_cache_for_append(cache_path)
        f.close()
        assert Path(cache_path).exists()
        with open(cache_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == CACHE_FIELDS

    def test_open_cache_for_append_no_duplicate_header(self, tmp_path):
        """Opening an existing file a second time must not write a second header."""
        cache_path = str(tmp_path / "cache.csv")
        f = open_cache_for_append(cache_path)
        f.close()
        f = open_cache_for_append(cache_path)
        f.close()
        with open(cache_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # Only the header row, no duplicate
        assert rows == [CACHE_FIELDS]

    def test_entries_written_during_build_hashes(self, tmp_path):
        """Each newly computed hash appears in the file before build_hashes returns."""
        imgs = [write_image(tmp_path / f"p{i}.jpg", color=("red","green","blue")[i])
                for i in range(3)]
        cache_path = str(tmp_path / "cache.csv")
        cache = {}
        cache_out = open_cache_for_append(cache_path)
        try:
            build_hashes([str(p) for p in imgs], cache, cache_out)
        finally:
            cache_out.close()

        loaded = load_cache(cache_path)
        assert len(loaded) == 3

    def test_cache_hit_does_not_append_row(self, tmp_path):
        """Files served from cache (no recompute) must not add a duplicate row."""
        p = write_image(tmp_path / "photo.jpg")
        cache_path = str(tmp_path / "cache.csv")

        # First run: 1 new hash written
        cache = {}
        cache_out = open_cache_for_append(cache_path)
        build_hashes([str(p)], cache, cache_out)
        cache_out.close()
        save_cache(cache_path, cache)   # compact

        # Second run: cache hit — file should still have exactly 1 data row
        cache2 = load_cache(cache_path)
        cache_out2 = open_cache_for_append(cache_path)
        _, updated, _, _ = build_hashes([str(p)], cache2, cache_out2)
        cache_out2.close()

        assert updated == 0
        with open(cache_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1   # no extra row appended

    def test_recovery_after_interrupt(self, tmp_path):
        """Hashes written before an interrupt are recoverable on the next run."""
        imgs = [write_image(tmp_path / f"img{i}.jpg") for i in range(4)]
        cache_path = str(tmp_path / "cache.csv")

        # Simulate processing only the first 2 files (interrupt before the rest)
        cache = {}
        cache_out = open_cache_for_append(cache_path)
        build_hashes([str(imgs[0]), str(imgs[1])], cache, cache_out)
        cache_out.close()
        # Do NOT call save_cache — simulates abrupt termination

        # Next run: load the partially written cache
        recovered = load_cache(cache_path)
        assert len(recovered) == 2

        # Resume: remaining 2 files should be computed, first 2 are cache hits
        cache_out2 = open_cache_for_append(cache_path)
        _, updated, _, _ = build_hashes([str(p) for p in imgs], recovered, cache_out2)
        cache_out2.close()

        assert updated == 2   # only the two that weren't processed before

    def test_compaction_removes_duplicate_rows(self, tmp_path):
        """save_cache after append mode produces a clean single-entry-per-file CSV."""
        p = write_image(tmp_path / "photo.jpg")
        cache_path = str(tmp_path / "cache.csv")

        # Two runs without compaction between them → file has 2 rows for the same file
        cache = {}
        cache_out = open_cache_for_append(cache_path)
        build_hashes([str(p)], cache, cache_out)
        cache_out.close()

        # Force a re-hash by clearing the in-memory cache (simulates stale detection)
        cache2 = {}
        # Corrupt mtime in the on-disk cache so second run sees it as stale
        loaded = load_cache(cache_path)
        key = next(iter(loaded))
        loaded[key]["mtime"] -= _TS_TOLERANCE + 10

        cache_out2 = open_cache_for_append(cache_path)
        build_hashes([str(p)], loaded, cache_out2)
        cache_out2.close()

        # File now has 2 data rows for the same path; compaction should fix that
        save_cache(cache_path, loaded)
        with open(cache_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1


# ── CLI: --cache argument resolution ──────────────────────────────────────────

SCRIPT = os.path.join(os.path.dirname(__file__), "idem.py")


def _run(args: list, tmp_path) -> subprocess.CompletedProcess:
    """Run idem.py as a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True, text=True, cwd=str(tmp_path),
    )


class TestCLICacheArg:
    """Integration tests for --cache directory/file resolution."""

    def test_no_cache_arg_creates_file_in_scanned_dir(self, tmp_path):
        """Default: cache file is created inside the __databases subdirectory."""
        write_image(tmp_path / "photo.jpg")
        result = _run([str(tmp_path)], tmp_path)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / DB_DIR / CACHE_FILENAME).exists()

    def test_cache_as_directory_places_file_inside_it(self, tmp_path):
        """--cache <dir> puts images_perceptual_hash_db.csv inside that directory."""
        write_image(tmp_path / "photo.jpg")
        cache_dir = tmp_path / "my_cache_dir"
        cache_dir.mkdir()
        result = _run([str(tmp_path), "--cache", str(cache_dir)], tmp_path)
        assert result.returncode == 0, result.stderr
        assert (cache_dir / CACHE_FILENAME).exists()

    def test_cache_as_new_file_path_is_created(self, tmp_path):
        """--cache <file> that does not yet exist is created by the script."""
        write_image(tmp_path / "photo.jpg")
        cache_file = tmp_path / "custom_cache.csv"
        assert not cache_file.exists()
        result = _run([str(tmp_path), "--cache", str(cache_file)], tmp_path)
        assert result.returncode == 0, result.stderr
        assert cache_file.exists()

    def test_cache_as_existing_file_path_is_reused(self, tmp_path):
        """--cache <file> that already exists is loaded and updated."""
        p = write_image(tmp_path / "photo.jpg")
        cache_file = tmp_path / "custom_cache.csv"

        # First run — populates the cache
        result1 = _run([str(tmp_path), "--cache", str(cache_file)], tmp_path)
        assert result1.returncode == 0, result1.stderr

        # Second run — should report 0 new/updated hashes (all from cache)
        result2 = _run([str(tmp_path), "--cache", str(cache_file)], tmp_path)
        assert result2.returncode == 0, result2.stderr
        assert "0 new  *  0 re-hashed" in result2.stdout


# ── _name_score ─────────────────────────────────────────────────────────────────

class TestNameScoreCameraNames:
    """Camera/app-generated filenames should score 0 (no meaningful English words)."""

    def test_google_pixel(self):
        """Google Pixel PXL_ prefix is a noise token; the name scores 0."""
        assert _name_score("PXL_20231015_123456789.jpg") == 0

    def test_generic_img(self):
        """Generic 'IMG_' prefix is a noise token; the name scores 0."""
        assert _name_score("IMG_001.jpg") == 0

    def test_sony_dsc(self):
        """Sony DSC_ prefix is a noise token; the name scores 0."""
        assert _name_score("DSC_0001.jpg") == 0

    def test_fuji_dscf(self):
        """Fuji DSCF prefix is a noise token; the name scores 0."""
        assert _name_score("DSCF0042.jpg") == 0

    def test_nikon_dscn(self):
        """Nikon DSCN prefix is a noise token; the name scores 0."""
        assert _name_score("DSCN0099.JPG") == 0

    def test_whatsapp(self):
        """WhatsApp WA-prefixed names are noise tokens; the name scores 0."""
        assert _name_score("WA0001234567-1.jpg") == 0

    def test_dcim_prefix(self):
        """DCIM_ prefix is a noise token; the name scores 0."""
        assert _name_score("DCIM_001.jpg") == 0

    def test_pure_digits(self):
        """A filename that is purely numeric (e.g. a date) scores 0."""
        assert _name_score("20231015.jpg") == 0

    def test_date_with_time(self):
        """A date+time-only filename (e.g. 20231015_153022) scores 0."""
        assert _name_score("20231015_153022.jpg") == 0

    def test_dcf_token(self):
        """DCF_ token is recognised as noise; the name scores 0."""
        assert _name_score("DCF_0001.jpg") == 0

    def test_whatsapp_token(self):
        """'WhatsApp' is a noise token; only digits remain after splitting, scoring 0."""
        # 'WhatsApp' is a noise token; only digits remain after splitting
        assert _name_score("WhatsApp_2023-10-15.jpg") == 0

    def test_images_token(self):
        """'Images' is a noise token; the name scores 0."""
        # 'Images' is a noise token
        assert _name_score("images_001.jpg") == 0

    def test_whatsapp_and_images_combined(self):
        """Both 'WhatsApp' and 'Images' tokens are noise; nothing meaningful remains."""
        # Both tokens are noise; nothing meaningful remains
        assert _name_score("WhatsApp_Images_2023-10-15.jpg") == 0

    def test_whatsapp_case_insensitive(self):
        """'WhatsApp' noise detection is case-insensitive (whatsapp/WHATSAPP both score 0)."""
        assert _name_score("whatsapp_2023.jpg") == 0
        assert _name_score("WHATSAPP_2023.jpg") == 0
        assert _name_score("WhatsApp_2023.jpg") == 0

    def test_images_case_insensitive(self):
        """'Images' noise detection is case-insensitive (images/IMAGES/Images all score 0)."""
        assert _name_score("images_001.jpg") == 0
        assert _name_score("IMAGES_001.jpg") == 0
        assert _name_score("Images_001.jpg") == 0


class TestNameScoreMeaningfulNames:
    """Filenames with real English words should score > 0."""

    def test_two_words_underscore(self):
        """Two meaningful words separated by underscore score their combined letter count."""
        # 'beach'(5) + 'vacation'(8) = 13
        assert _name_score("beach_vacation.jpg") == 13

    def test_two_words_with_year(self):
        """A year suffix is stripped; only the English words contribute to the score."""
        # year is stripped; 'birthday'(8) + 'party'(5) = 13
        assert _name_score("birthday_party_2023.jpg") == 13

    def test_hyphen_separated(self):
        """Words separated by hyphens are each scored individually."""
        # 'new'(3) + 'year'(4) + 'eve'(3) = 10
        assert _name_score("new-year-eve.jpg") == 10

    def test_no_separator(self):
        """A single token with no separator is scored by its total character length."""
        # single CamelCase token, no noise match: 'BeachSunset'(11)
        assert _name_score("BeachSunset.jpg") == 11

    def test_date_prefix_meaningful_suffix(self):
        """A date prefix is stripped; only the meaningful suffix contributes to the score."""
        # digits/underscore split away; only 'birthday'(8) remains
        assert _name_score("20231015_birthday.jpg") == 8

    def test_long_descriptive_name(self):
        """A multi-word descriptive filename accumulates scores for all words."""
        # 'beautiful'(9)+'sunset'(6)+'at'(2)+'beach'(5) = 22
        assert _name_score("beautiful_sunset_at_beach.jpg") == 22

    def test_space_separated(self):
        """Spaces serve as word separators in the same way underscores do."""
        # spaces are separators; 'family'(6)+'reunion'(7) = 13
        assert _name_score("family reunion.jpg") == 13

    def test_single_meaningful_word(self):
        """A single meaningful word scores its character length."""
        assert _name_score("sunset.jpg") == 6

    def test_uppercase_extension_ignored(self):
        """The file extension is stripped regardless of case before scoring."""
        # extension is stripped; 'beach'(5)+'vacation'(8) = 13
        assert _name_score("beach_vacation.JPEG") == 13


class TestNameScoreComparisons:
    """A meaningful name always outscores a camera name."""

    @pytest.mark.parametrize("camera,meaningful", [
        ("PXL_20231015_123456789.jpg", "beach_vacation.jpg"),
        ("IMG_001.jpg",                "birthday_party.jpg"),
        ("DSC_0042.jpg",               "christmas_morning.jpg"),
        ("WA0001234567-1.jpg",         "holiday_trip.jpg"),
        ("20231015_153022.jpg",        "sunset.jpg"),
    ])
    def test_meaningful_beats_camera(self, camera, meaningful):
        """Any descriptive filename scores higher than any camera-generated name."""
        assert _name_score(meaningful) > _name_score(camera)

    def test_more_words_beats_fewer(self):
        """A more descriptive (longer) name always outscores a shorter one."""
        assert _name_score("beautiful_sunset_at_beach.jpg") > _name_score("sunset.jpg")

    def test_camera_tie(self):
        """Two camera-style names both score 0 and are therefore tied."""
        # Two camera-style names both score 0
        assert _name_score("PXL_001.jpg") == _name_score("IMG_002.jpg") == 0

    def test_noise_token_case_insensitive(self):
        """Noise tokens are matched case-insensitively (lowercase 'pxl' treated same as 'PXL')."""
        # Lowercase 'pxl' should be treated as noise just like 'PXL'
        assert _name_score("pxl_20231015_001.jpg") == 0


# ── _folder_score ────────────────────────────────────────────────────────────────

class TestFolderScoreDateFolders:
    """Folders that are purely dates or numbers should score 0."""

    def test_iso_date(self):
        """ISO-format date folders (YYYY-MM-DD) score 0."""
        assert _folder_score("/Photos/2023-12-25") == 0

    def test_year_only(self):
        """Year-only folder names score 0."""
        assert _folder_score("/Photos/2023") == 0

    def test_year_month(self):
        """Year-month folder names (YYYY-MM) score 0."""
        assert _folder_score("/Photos/2023-12") == 0

    def test_underscore_date(self):
        """Underscore-separated date folders (YYYY_MM_DD) score 0."""
        assert _folder_score("/Photos/2023_12_25") == 0

    def test_pure_number(self):
        """Purely numeric folder names score 0."""
        assert _folder_score("/camera/1234") == 0


class TestFolderScoreMeaningfulFolders:
    """Folders with English words should score > 0."""

    def test_two_words(self):
        """A two-word folder name scores the total character count of its words."""
        # 'BeachVacation' = 13
        assert _folder_score("/Photos/Beach Vacation") == 13

    def test_word_with_year(self):
        """A year suffix is stripped; only the English word contributes to the folder score."""
        # digits stripped; 'Christmas'(9) remains
        assert _folder_score("/Photos/Christmas 2023") == 9

    def test_underscore_separated(self):
        """Underscores act as word separators in folder names."""
        # underscores stripped; 'familyreunion'(13)
        assert _folder_score("/2023/family_reunion") == 13

    def test_only_last_component_scored(self):
        """Only the last path component (immediate parent folder) is scored."""
        # Deep path vs shallow path with same last component must score equally
        assert _folder_score("/very/long/path/Christmas Party") == _folder_score("/Christmas Party")

    def test_camera_roll(self):
        """'Camera Roll' is a recognisable folder name, scoring its letter count."""
        # 'CameraRoll' = 10
        assert _folder_score("/Phone/Camera Roll") == 10

    def test_letters_only_folder(self):
        """An all-letter folder name (not a noise token) scores its character count."""
        # 'DCIM' has 4 alpha chars; not stripped (folder score doesn't filter noise tokens)
        assert _folder_score("/DCIM") == 4

    def test_mixed_alpha_digits(self):
        """Digits within a folder name are stripped before scoring the remaining letters."""
        # '100MEDIA': strip digits → 'MEDIA'(5)
        assert _folder_score("/camera/100MEDIA") == 5


class TestFolderScoreComparisons:
    """Descriptive folders outrank date/numeric ones."""

    @pytest.mark.parametrize("date_folder,named_folder", [
        ("/2023-12-25",  "/Christmas Morning"),
        ("/2023",        "/Beach Vacation"),
        ("/2023-08",     "/Summer Holidays"),
        ("/20231015",    "/Birthday Party"),
    ])
    def test_named_beats_date(self, date_folder, named_folder):
        """Descriptive folder names always score higher than date/numeric ones."""
        assert _folder_score(named_folder) > _folder_score(date_folder)


# ── _smart_defaults — helpers ────────────────────────────────────────────────────

def _file(path, name, dir_, size):
    return {"path": path, "name": name, "dir": dir_, "size": size}


# ── _smart_defaults — keep selection ─────────────────────────────────────────────

class TestSmartDefaultsKeep:
    """Index-0 (the largest file) is always kept."""

    def test_keeps_largest(self):
        """The largest file (index 0 after size-sort) is always the kept file."""
        files = [
            _file("/a/big.jpg", "big.jpg", "/a", 5_000_000),
            _file("/b/small.jpg", "small.jpg", "/b", 1_000_000),
        ]
        keep, _, _ = _smart_defaults(files)
        assert keep == "/a/big.jpg"

    def test_keeps_index_zero_even_if_camera_name(self):
        """The keep choice is always index 0 regardless of filename quality."""
        files = [
            _file("/a/PXL_001.jpg", "PXL_001.jpg", "/a", 8_000_000),
            _file("/b/beach_vacation.jpg", "beach_vacation.jpg", "/b", 2_000_000),
        ]
        keep, _, _ = _smart_defaults(files)
        assert keep == "/a/PXL_001.jpg"

    def test_keeps_index_zero_three_files(self):
        """With three files, index 0 is still kept regardless of the other names."""
        files = [
            _file("/a/img.jpg", "img.jpg", "/a", 10_000_000),
            _file("/b/medium.jpg", "medium.jpg", "/b", 5_000_000),
            _file("/c/small.jpg", "small.jpg", "/c", 1_000_000),
        ]
        keep, _, _ = _smart_defaults(files)
        assert keep == "/a/img.jpg"


# ── _smart_defaults — rename source ──────────────────────────────────────────────

class TestSmartDefaultsRename:
    """Auto-selects the non-kept file with the most meaningful name."""

    def test_selects_meaningful_over_camera(self):
        """A file with a meaningful name is selected as the rename source over a camera name."""
        files = [
            _file("/a/PXL_20231015_123.jpg", "PXL_20231015_123.jpg", "/a", 5_000_000),
            _file("/b/beach_vacation.jpg",   "beach_vacation.jpg",   "/b", 1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src == "/b/beach_vacation.jpg"

    def test_no_rename_when_kept_already_has_good_name(self):
        """No rename is suggested when the kept file already has a better name than all others."""
        # Kept file: 'christmas_morning' scores 16; non-kept IMG_001 scores 0
        files = [
            _file("/a/christmas_morning.jpg", "christmas_morning.jpg", "/a", 5_000_000),
            _file("/b/IMG_001.jpg",           "IMG_001.jpg",           "/b", 1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src is None

    def test_no_rename_when_all_camera_names(self):
        """When all files have camera-style names (score 0), no rename is suggested."""
        files = [
            _file("/a/PXL_001.jpg", "PXL_001.jpg", "/a", 5_000_000),
            _file("/b/IMG_002.jpg", "IMG_002.jpg", "/b", 1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src is None

    def test_picks_best_name_among_multiple(self):
        """The file with the highest name score is selected as the rename source."""
        # File 1 (sunset, 6 pts) loses to file 2 (beautiful_sunset_at_beach, 22 pts)
        files = [
            _file("/a/PXL_001.jpg",                  "PXL_001.jpg",                  "/a", 10_000_000),
            _file("/b/sunset.jpg",                   "sunset.jpg",                   "/b",  3_000_000),
            _file("/c/beautiful_sunset_at_beach.jpg","beautiful_sunset_at_beach.jpg", "/c",  1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src == "/c/beautiful_sunset_at_beach.jpg"

    def test_tie_means_no_rename(self):
        """When kept and non-kept files have equal name scores, no rename is suggested."""
        # Kept file and only non-kept file have equal name scores → no change
        # sunset(6) vs clouds(6) — tied, so no rename
        files = [
            _file("/a/sunset.jpg", "sunset.jpg", "/a", 5_000_000),
            _file("/b/clouds.jpg", "clouds.jpg", "/b", 1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src is None

    def test_noise_token_case_insensitive(self):
        """Noise-token detection is case-insensitive for rename selection."""
        # 'pxl_001' should be treated as noise (lowercase)
        files = [
            _file("/a/pxl_20231015_001.jpg", "pxl_20231015_001.jpg", "/a", 5_000_000),
            _file("/b/beach_vacation.jpg",   "beach_vacation.jpg",   "/b", 1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src == "/b/beach_vacation.jpg"


# ── _smart_defaults — folder source ──────────────────────────────────────────────

class TestSmartDefaultsFolder:
    """Auto-selects the non-kept file whose immediate parent folder is most meaningful."""

    def test_selects_named_folder_over_date(self):
        """A named folder is selected as folder source over a date folder."""
        files = [
            _file("/2023-12-25/PXL_001.jpg",        "PXL_001.jpg",  "/2023-12-25",       5_000_000),
            _file("/Christmas Morning/beach.jpg",    "beach.jpg",    "/Christmas Morning", 1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files)
        assert folder_src == "/Christmas Morning/beach.jpg"

    def test_no_folder_when_kept_already_in_good_folder(self):
        """No folder move is suggested when the kept file is already in the best-named folder."""
        # Kept: 'Beach Vacation'(13) > non-kept '2023-08'(0)
        files = [
            _file("/Beach Vacation/big.jpg", "big.jpg", "/Beach Vacation", 5_000_000),
            _file("/2023-08/small.jpg",      "small.jpg", "/2023-08",      1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files)
        assert folder_src is None

    def test_no_folder_when_all_date_folders(self):
        """When all files are in date folders, no folder move is suggested."""
        files = [
            _file("/2023-12/PXL_001.jpg", "PXL_001.jpg", "/2023-12", 5_000_000),
            _file("/2023-11/IMG_002.jpg", "IMG_002.jpg", "/2023-11", 1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files)
        assert folder_src is None

    def test_picks_best_folder_among_multiple(self):
        """The file in the folder with the highest score is chosen as the folder source."""
        # 'Trips'(5) < 'Beach Vacation 2023'(13)
        files = [
            _file("/2023/PXL.jpg",              "PXL.jpg", "/2023",              10_000_000),
            _file("/Trips/a.jpg",               "a.jpg",   "/Trips",              3_000_000),
            _file("/Beach Vacation 2023/b.jpg", "b.jpg",   "/Beach Vacation 2023",1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files)
        assert folder_src == "/Beach Vacation 2023/b.jpg"

    def test_folder_source_never_equals_keep(self):
        """The folder source is always a non-kept file; it is never the kept file itself."""
        # Kept file is in the best folder; no non-kept file can beat it
        files = [
            _file("/Christmas Party/big.jpg", "big.jpg",   "/Christmas Party", 5_000_000),
            _file("/2023/small.jpg",          "small.jpg", "/2023",            1_000_000),
        ]
        keep, _, folder_src = _smart_defaults(files)
        assert folder_src is None
        assert folder_src != keep


# ── _smart_defaults — combined scenarios ─────────────────────────────────────────

class TestSmartDefaultsCombined:
    """Name and folder sources can be selected from different files."""

    def test_best_name_and_folder_from_different_files(self):
        """The best name and best folder may come from different non-kept files."""
        # File 1 has the best name; file 2 has the best folder
        files = [
            _file("/2023/PXL_001.jpg",           "PXL_001.jpg",        "/2023",           10_000_000),
            _file("/2023/beach_vacation.jpg",     "beach_vacation.jpg", "/2023",            3_000_000),
            _file("/Summer Holidays/IMG_002.jpg", "IMG_002.jpg",        "/Summer Holidays", 1_000_000),
        ]
        keep, rename_src, folder_src = _smart_defaults(files)
        assert keep == "/2023/PXL_001.jpg"
        assert rename_src == "/2023/beach_vacation.jpg"    # score 13 > 0
        assert folder_src == "/Summer Holidays/IMG_002.jpg"  # score 14 > 0

    def test_best_name_and_folder_from_same_file(self):
        """A single non-kept file can supply both the best name and the best folder."""
        # One non-kept file has both the best name and the best folder
        files = [
            _file("/2023/PXL_001.jpg",       "PXL_001.jpg", "/2023",      5_000_000),
            _file("/Beach Trip/holiday.jpg", "holiday.jpg", "/Beach Trip", 1_000_000),
        ]
        keep, rename_src, folder_src = _smart_defaults(files)
        assert keep == "/2023/PXL_001.jpg"
        assert rename_src == "/Beach Trip/holiday.jpg"   # 'holiday'(7) > 0
        assert folder_src == "/Beach Trip/holiday.jpg"   # 'BeachTrip'(9) > 0

    def test_no_defaults_when_all_noise(self):
        """No rename or folder move is suggested when all names and folders are noise."""
        # All camera names in date folders → nothing auto-selected
        files = [
            _file("/2023-12/PXL_001.jpg", "PXL_001.jpg", "/2023-12", 5_000_000),
            _file("/2023-11/IMG_002.jpg", "IMG_002.jpg", "/2023-11", 1_000_000),
        ]
        keep, rename_src, folder_src = _smart_defaults(files)
        assert keep == "/2023-12/PXL_001.jpg"
        assert rename_src is None
        assert folder_src is None

    def test_realistic_pixel_vs_whatsapp(self):
        """A Pixel original vs a WhatsApp copy: keep the Pixel, move from the named 'Media' folder."""
        # Pixel photo (large, date folder) vs WhatsApp copy (small, 'Media' folder)
        # WA name scores 0; 'Media'(5) > '2023-08'(0) → folder changes
        files = [
            _file("C:/Photos/2023-08/PXL_20230815_183042.jpg",
                  "PXL_20230815_183042.jpg", "C:/Photos/2023-08", 4_500_000),
            _file("C:/WhatsApp/Media/WA00012345-1.jpg",
                  "WA00012345-1.jpg",         "C:/WhatsApp/Media",   800_000),
        ]
        keep, rename_src, folder_src = _smart_defaults(files)
        assert keep == "C:/Photos/2023-08/PXL_20230815_183042.jpg"
        assert rename_src is None                                   # WA name is also noise
        assert folder_src == "C:/WhatsApp/Media/WA00012345-1.jpg"  # 'Media'(5) > '2023-08'(0)

    def test_realistic_pixel_vs_edited_copy(self):
        """A Pixel original vs a descriptively named edit: keep the Pixel, rename and move from the edit."""
        # Large original (camera name, date folder) vs small copy with meaningful name+folder
        files = [
            _file("C:/Photos/2023/PXL_20230815_183042.jpg",
                  "PXL_20230815_183042.jpg", "C:/Photos/2023",      4_500_000),
            _file("C:/Shared/Beach Trip/sunset_at_goa.jpg",
                  "sunset_at_goa.jpg",        "C:/Shared/Beach Trip",  600_000),
        ]
        keep, rename_src, folder_src = _smart_defaults(files)
        assert keep == "C:/Photos/2023/PXL_20230815_183042.jpg"
        assert rename_src == "C:/Shared/Beach Trip/sunset_at_goa.jpg"  # 'sunsetatgoa'(11) > 0
        assert folder_src == "C:/Shared/Beach Trip/sunset_at_goa.jpg"  # 'BeachTrip'(9) > 0



# ── extra_noise / --ignore ────────────────────────────────────────────────────────

class TestExtraNoiseNameScore:
    """--ignore words are stripped from filename scoring."""

    def test_single_ignored_word_zeros_name(self):
        """Ignoring one word removes it from scoring, leaving only the remaining tokens."""
        # "vacation" is an exact token of "beach_vacation" → that token is stripped, "beach" (5) remains
        assert _name_score("beach_vacation.jpg") == 13
        assert _name_score("beach_vacation.jpg", extra_noise=["vacation"]) == 5

    def test_ignored_word_alone_scores_zero(self):
        """A filename consisting solely of an ignored word scores 0."""
        assert _name_score("vacation.jpg", extra_noise=["vacation"]) == 0

    def test_multiple_ignored_words_either_hits(self):
        """Multiple ignored words each strip their matching tokens, potentially zeroing the score."""
        # Either word matches → 0
        assert _name_score("beach_vacation.jpg", extra_noise=["beach", "vacation"]) == 0

    def test_ignore_is_case_insensitive(self):
        """Name ignore matching is case-insensitive regardless of token or pattern case."""
        assert _name_score("Beach_Vacation.jpg", extra_noise=["BEACH", "VACATION"]) == 0
        assert _name_score("Beach_Vacation.jpg", extra_noise=["beach", "vacation"]) == 0
        assert _name_score("Beach_Vacation.jpg", extra_noise=["Beach", "Vacation"]) == 0

    def test_pattern_not_present_does_not_affect_score(self):
        """An ignore pattern not present in the filename leaves the score unchanged."""
        # "trip" is not in "beach_vacation" → score unchanged
        assert _name_score("beach_vacation.jpg", extra_noise=["trip"]) == 13

    def test_ignore_zeros_name_containing_pattern(self):
        """Stripping an ignored token lowers but does not necessarily zero the score."""
        # "vacation" token stripped from "beach_vacation_sunset" → "beach"(5)+"sunset"(6) = 11
        assert _name_score("beach_vacation_sunset.jpg", extra_noise=["vacation"]) == 11

    def test_empty_extra_noise_unchanged(self):
        """An empty ignore list does not change the name score."""
        assert _name_score("beach_vacation.jpg", extra_noise=[]) == 13

    def test_hyphenated_pattern_not_a_substring(self):
        """A hyphenated ignore pattern does not match tokens joined by underscores."""
        # "beach-vacation" is NOT a substring of "beach_vacation" → score unchanged
        assert _name_score("beach_vacation.jpg", extra_noise=["beach-vacation"]) == 13

    def test_underscore_pattern(self):
        """An underscore-joined ignore pattern matches the full underscore token."""
        # "beach_vacation" IS a substring of "beach_vacation" → score 0
        assert _name_score("beach_vacation.jpg", extra_noise=["beach_vacation"]) == 0

    def test_space_pattern(self):
        """A space-joined ignore pattern matches the full space-separated token."""
        assert _name_score("beach vacation.jpg", extra_noise=["beach vacation"]) == 0


class TestExtraNoiseFolderScore:
    """--ignore words are stripped from folder scoring (exact token or exact whole-name match)."""

    def test_single_ignored_word_strips_token(self):
        """Ignoring one word removes it from the folder name score, leaving the rest."""
        # "vacation" is an exact token of "Beach Vacation" → stripped, "Beach" (5) remains
        assert _folder_score("/Photos/Beach Vacation") == 13
        assert _folder_score("/Photos/Beach Vacation", extra_noise=["vacation"]) == 5

    def test_ignored_word_alone_scores_minus_one(self):
        """A folder whose entire name is an ignored word scores -1 (demoted below date folders)."""
        # Entire folder component exactly equals noise word → scores -1 (demoted)
        assert _folder_score("/Vacation", extra_noise=["vacation"]) == -1

    def test_multiple_ignored_words_both_stripped(self):
        """Multiple ignored words each strip their matching tokens from the folder name."""
        # Both tokens stripped → empty → 0
        assert _folder_score("/Beach Vacation", extra_noise=["beach", "vacation"]) == 0

    def test_ignore_is_case_insensitive(self):
        """Folder ignore matching is case-insensitive regardless of token or pattern case."""
        assert _folder_score("/Beach Vacation", extra_noise=["BEACH", "VACATION"]) == 0
        assert _folder_score("/Beach Vacation", extra_noise=["Beach", "Vacation"]) == 0

    def test_pattern_not_present_does_not_affect_score(self):
        """An ignore pattern not present in the folder name leaves the score unchanged."""
        # "trip" is not in "DCIM" → score unchanged
        assert _folder_score("/DCIM", extra_noise=["vacation"]) == 4

    def test_empty_extra_noise_unchanged(self):
        """An empty ignore list does not change the folder score."""
        assert _folder_score("/Beach Vacation", extra_noise=[]) == 13

    def test_hyphenated_pattern_not_a_substring(self):
        """A hyphenated pattern does not match space-separated or underscore-separated folder tokens."""
        # "beach-vacation" does not match token "Beach" or "Vacation" → score unchanged
        assert _folder_score("/Beach Vacation", extra_noise=["beach-vacation"]) == 13

    def test_underscore_pattern(self):
        """An underscore-joined pattern matching the whole folder component scores -1."""
        # "beach_vacation" exactly equals the whole folder component → scores -1
        assert _folder_score("/Beach_Vacation", extra_noise=["beach_vacation"]) == -1

    def test_space_pattern(self):
        """A space-joined pattern matching the whole folder component scores -1."""
        # "beach vacation" exactly equals the whole folder component → scores -1
        assert _folder_score("/Beach Vacation", extra_noise=["beach vacation"]) == -1


class TestExtraNoiseSmartDefaults:
    """--ignore flows through _smart_defaults selection logic."""

    def test_ignore_demotes_filename_so_no_rename_selected(self):
        """Ignoring all words in a filename drops its score to 0, preventing rename selection."""
        # Without ignore: 'beach_vacation'(13) wins over PXL(0) → rename selected
        files = [
            _file("/a/PXL_001.jpg",         "PXL_001.jpg",       "/a", 5_000_000),
            _file("/b/beach_vacation.jpg",   "beach_vacation.jpg", "/b", 1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src == "/b/beach_vacation.jpg"

        # With both words ignored: score drops to 0, no rename
        _, rename_src, _ = _smart_defaults(files, extra_noise=["beach", "vacation"])
        assert rename_src is None

    def test_ignore_demotes_folder_so_no_folder_selected(self):
        """Ignoring all words in a folder name drops its score to 0, preventing folder selection."""
        # Without ignore: 'Beach Trip'(9) wins over '2023'(0) → folder selected
        files = [
            _file("/2023/PXL_001.jpg",     "PXL_001.jpg", "/2023",      5_000_000),
            _file("/Beach Trip/a.jpg",     "a.jpg",        "/Beach Trip", 1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files)
        assert folder_src == "/Beach Trip/a.jpg"

        # With both words ignored: both tokens stripped → score 0, same as date folder → no move
        _, _, folder_src = _smart_defaults(files, extra_noise=["beach", "trip"])
        assert folder_src is None

    def test_ignore_selects_different_best_name(self):
        """Ignoring a word can shift which non-kept file has the highest name score."""
        # File 1: 'sunset_beach'(11); file 2: 'sunset_vacation'(14)
        # Ignoring 'vacation' makes file 1 win
        files = [
            _file("/a/PXL.jpg",              "PXL.jpg",             "/a", 10_000_000),
            _file("/b/sunset_beach.jpg",     "sunset_beach.jpg",    "/b",  3_000_000),
            _file("/c/sunset_vacation.jpg",  "sunset_vacation.jpg", "/c",  1_000_000),
        ]
        _, rename_src, _ = _smart_defaults(files)
        assert rename_src == "/c/sunset_vacation.jpg"  # 14 > 11 without ignore

        _, rename_src, _ = _smart_defaults(files, extra_noise=["vacation"])
        assert rename_src == "/b/sunset_beach.jpg"     # 11 > 6 (only 'sunset' left in file 2)

    def test_ignore_does_not_affect_keep(self):
        """extra_noise never affects which file is kept (always index 0 regardless of noise)."""
        # extra_noise never changes which file is kept (always index 0)
        files = [
            _file("/a/vacation.jpg", "vacation.jpg", "/Vacation", 5_000_000),
            _file("/b/beach.jpg",    "beach.jpg",    "/Beach",    1_000_000),
        ]
        keep, _, _ = _smart_defaults(files, extra_noise=["vacation", "beach"])
        assert keep == "/a/vacation.jpg"

    def test_ignored_kept_folder_defers_to_non_ignored_folder(self):
        """A kept file in an ignored folder (score -1) loses to a date folder (score 0), triggering a move."""
        # Kept file is in a folder whose name exactly equals the noise word (scores -1).
        # The date folder scores 0 → 0 > -1 → move to date folder is suggested.
        files = [
            _file("/Vacation/big.jpg",  "big.jpg",  "/Vacation", 5_000_000),
            _file("/2023-08/small.jpg", "small.jpg", "/2023-08",  1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files, extra_noise=["vacation"])
        assert folder_src == "/2023-08/small.jpg"

    def test_all_ignored_folders_no_folder_move(self):
        """When every replica is in an ignored folder, no folder move is suggested."""
        # Every replica is in an ignored folder → destination stays as-is.
        files = [
            _file("/Vacation/big.jpg",  "big.jpg",  "/Vacation", 5_000_000),
            _file("/Holiday/small.jpg", "small.jpg", "/Holiday",  1_000_000),
        ]
        _, _, folder_src = _smart_defaults(files, extra_noise=["vacation", "holiday"])
        assert folder_src is None


# ── _groups_to_json — smart defaults embedding ───────────────────────────────────

class TestGroupsToJson:
    """Verify _groups_to_json embeds server-computed smart defaults per group."""

    def _make_group(self, *entries):
        """entries: (abs_path, size_bytes) tuples."""
        return list(entries)

    def test_keep_default_is_first_file(self):
        """keep_default is always the first (largest) file in the group."""
        from idem import _groups_to_json
        groups = [[(("/a/big.jpg"), 5_000_000), ("/b/small.jpg", 1_000_000)]]
        result = _groups_to_json(groups, "/")
        assert result[0]["keep_default"] == "/a/big.jpg"

    def test_rename_src_none_when_first_name_wins(self):
        """rename_src is None when the first file already has the best name."""
        # First file has the most meaningful name → no rename suggested
        from idem import _groups_to_json
        groups = [[(("/a/sunset_beach.jpg"), 5_000_000), ("/b/IMG_001.jpg", 1_000_000)]]
        result = _groups_to_json(groups, "/")
        assert result[0]["rename_src"] is None

    def test_rename_src_set_when_other_name_wins(self):
        """rename_src is set to the file with the better name when the kept file's name is weaker."""
        from idem import _groups_to_json
        groups = [[(("/a/IMG_001.jpg"), 5_000_000), ("/b/sunset_beach.jpg", 1_000_000)]]
        result = _groups_to_json(groups, "/")
        assert result[0]["rename_src"] == "/b/sunset_beach.jpg"

    def test_folder_src_set_when_other_folder_wins(self):
        """folder_src is set to the file in the most descriptively named folder."""
        from idem import _groups_to_json
        groups = [[(("/2023/IMG_001.jpg"), 5_000_000), ("/Beach Trip/a.jpg", 1_000_000)]]
        result = _groups_to_json(groups, "/")
        assert result[0]["folder_src"] == "/Beach Trip/a.jpg"

    def test_ignore_flows_through_to_smart_defaults(self):
        """The ignore list is forwarded to _smart_defaults, affecting folder_src selection."""
        # Without ignore, 'Beach Trip' folder wins → folder_src set.
        # With 'beach' and 'trip' ignored, both tokens stripped → score 0, same as kept → folder_src None.
        from idem import _groups_to_json
        groups = [[(("/2023/IMG_001.jpg"), 5_000_000), ("/Beach Trip/a.jpg", 1_000_000)]]
        without_ignore = _groups_to_json(groups, "/")
        assert without_ignore[0]["folder_src"] == "/Beach Trip/a.jpg"
        with_ignore = _groups_to_json(groups, "/", ignore=("beach", "trip"))
        assert with_ignore[0]["folder_src"] is None

    def test_ignore_case_insensitive(self):
        """Ignore matching in _groups_to_json is case-insensitive."""
        from idem import _groups_to_json
        groups = [[(("/2023/IMG_001.jpg"), 5_000_000), ("/Beach Trip/a.jpg", 1_000_000)]]
        result = _groups_to_json(groups, "/", ignore=("BEACH", "TRIP"))
        assert result[0]["folder_src"] is None

    def test_multiple_groups_each_get_defaults(self):
        """Each group in the output independently receives its own smart defaults."""
        from idem import _groups_to_json
        groups = [
            [("/a/IMG_001.jpg", 5_000_000), ("/b/sunset.jpg", 1_000_000)],
            [("/c/IMG_002.jpg", 3_000_000), ("/d/holiday.jpg", 500_000)],
        ]
        result = _groups_to_json(groups, "/")
        assert result[0]["rename_src"] == "/b/sunset.jpg"
        assert result[1]["rename_src"] == "/d/holiday.jpg"

    def test_fields_present_in_each_group(self):
        """Every group dict contains the required keys: keep_default, rename_src, folder_src, files."""
        from idem import _groups_to_json
        groups = [[("/a/x.jpg", 1_000_000), ("/b/y.jpg", 500_000)]]
        result = _groups_to_json(groups, "/")
        g = result[0]
        assert "keep_default" in g
        assert "rename_src" in g
        assert "folder_src" in g
        assert "files" in g


# ── CLI --ignore wiring ───────────────────────────────────────────────────────────

class TestCLIIgnoreArg:
    """--ignore reaches launch_review_ui (verified via CLI invocation)."""

    def test_ignore_arg_accepted(self, tmp_path):
        """--ignore flag is a valid argument and doesn't cause a CLI error."""
        write_image(tmp_path / "a.jpg")
        result = _run([str(tmp_path), "--ignore", "vacation"], tmp_path)
        assert result.returncode == 0, result.stderr

    def test_ignore_repeatable(self, tmp_path):
        """--ignore can appear multiple times."""
        write_image(tmp_path / "a.jpg")
        result = _run(
            [str(tmp_path), "--ignore", "vacation", "--ignore", "beach"],
            tmp_path,
        )
        assert result.returncode == 0, result.stderr


# ── CLI --limit wiring ────────────────────────────────────────────────────────────

class TestCLILimitArg:
    """--limit restricts the number of duplicate groups processed in review mode."""

    def test_limit_arg_accepted(self, tmp_path):
        """--limit is a valid argument and the script exits cleanly."""
        write_image(tmp_path / "a.jpg")
        result = _run([str(tmp_path), "--limit", "5"], tmp_path)
        assert result.returncode == 0, result.stderr

    def test_limit_zero_rejected(self, tmp_path):
        """--limit 0 is invalid and should exit with a non-zero code."""
        write_image(tmp_path / "a.jpg")
        result = _run([str(tmp_path), "--limit", "0"], tmp_path)
        assert result.returncode != 0
        assert "limit" in result.stderr.lower()

    def test_limit_negative_rejected(self, tmp_path):
        """--limit -1 is invalid and should exit with a non-zero code."""
        write_image(tmp_path / "a.jpg")
        result = _run([str(tmp_path), "--limit", "-1"], tmp_path)
        assert result.returncode != 0

    def test_limit_truncates_output(self, tmp_path):
        """With --limit 1, only 1 group appears in the report even when more exist."""
        # Create two pairs of visually distinct images that form 2 separate groups.
        # Solid-color images share the same pHash (DC-only signal), so use stripe
        # patterns with real frequency content to get distinct hashes.
        def write_striped(path, horizontal: bool):
            img = Image.new("L", (64, 64), 0)
            px = img.load()
            for y in range(64):
                for x in range(64):
                    stripe = (y // 8) if horizontal else (x // 8)
                    px[x, y] = 255 if stripe % 2 == 0 else 0
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            path.write_bytes(buf.getvalue())

        # Group 1: two copies of horizontal-stripe image
        write_striped(tmp_path / "h1.png", horizontal=True)
        write_striped(tmp_path / "h2.png", horizontal=True)
        # Group 2: two copies of vertical-stripe image
        write_striped(tmp_path / "v1.png", horizontal=False)
        write_striped(tmp_path / "v2.png", horizontal=False)

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        def run_script(*extra):
            return subprocess.run(
                [sys.executable, SCRIPT, str(tmp_path), "--threshold", "0"] + list(extra),
                capture_output=True, encoding="utf-8", cwd=str(tmp_path), env=env,
            )

        result_all   = run_script()
        result_limit = run_script("--limit", "1")
        assert result_all.returncode   == 0, result_all.stderr
        assert result_limit.returncode == 0, result_limit.stderr
        assert "Limiting to first" in result_limit.stdout
        assert result_all.stdout.count("Group ") == 2
        assert result_limit.stdout.count("Group ") == 1

    def test_limit_larger_than_groups_is_fine(self, tmp_path):
        """--limit larger than the number of groups processes all groups silently."""
        write_image(tmp_path / "a.jpg")
        result = _run([str(tmp_path), "--limit", "9999"], tmp_path)
        assert result.returncode == 0, result.stderr
        assert "Limiting" not in result.stdout


# ── _resolve_transform ─────────────────────────────────────────────────────────

class TestResolveTransform:
    """Unit tests for _resolve_transform(r, keep_set, dir_resolved).

    Returns (src, dst) on success, None for a no-op, or an error string.
    """

    def _make(self, tmp_path, filename="img.jpg"):
        """Write a real file and return its absolute path string."""
        p = tmp_path / filename
        p.write_bytes(b"x")
        return str(p)

    # ── path validation ────────────────────────────────────────────────────────

    def test_path_not_in_keep_set_is_error(self, tmp_path):
        """Attempting to transform a path not in the keep set returns an error string."""
        path = self._make(tmp_path)
        result = _resolve_transform(
            {"path": path, "new_name": "other.jpg"},
            keep_set=set(),
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "non-kept" in result

    def test_path_outside_dir_resolved_is_error(self, tmp_path):
        """A path outside the scan directory returns an error string."""
        outside = tmp_path.parent / "outside.jpg"
        outside.write_bytes(b"x")
        keep = str(outside)
        result = _resolve_transform(
            {"path": keep, "new_name": "renamed.jpg"},
            keep_set={keep},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "Invalid path" in result

    def test_file_not_on_disk_is_error(self, tmp_path):
        """A path that does not exist on disk returns an error string."""
        missing = str(tmp_path / "ghost.jpg")
        result = _resolve_transform(
            {"path": missing, "new_name": "other.jpg"},
            keep_set={missing},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "File not found" in result

    def test_non_string_path_is_error(self, tmp_path):
        """A non-string path value is rejected with an error string."""
        result = _resolve_transform(
            {"path": 42, "new_name": "other.jpg"},
            keep_set={42},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)

    # ── new_name validation ────────────────────────────────────────────────────

    def test_new_name_strips_directory_components(self, tmp_path):
        """Path-traversal attempt in new_name: only the filename part is kept."""
        path = self._make(tmp_path)
        src, dst = _resolve_transform(
            {"path": path, "new_name": "../../evil.jpg"},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert dst.name == "evil.jpg"
        assert dst.parent == tmp_path.resolve()

    def test_new_name_that_reduces_to_empty_is_error(self, tmp_path):
        """A new_name whose Path.name is '' (e.g. '/') is rejected."""
        path = self._make(tmp_path)
        result = _resolve_transform(
            {"path": path, "new_name": "/"},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "Invalid new name" in result

    def test_non_string_new_name_treated_as_absent(self, tmp_path):
        """A non-string new_name is coerced to '' and ignored (no rename)."""
        path = self._make(tmp_path, "keep.jpg")
        # With no new_name and no target_dir the filename is unchanged → no-op.
        result = _resolve_transform(
            {"path": path, "new_name": 99},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert result is None

    # ── target_dir validation ──────────────────────────────────────────────────

    def test_target_dir_outside_dir_resolved_is_error(self, tmp_path):
        """A target directory outside the scan root returns an error string."""
        path = self._make(tmp_path)
        result = _resolve_transform(
            {"path": path, "target_dir": str(tmp_path.parent)},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "Invalid target folder" in result

    def test_target_dir_does_not_exist_is_error(self, tmp_path):
        """A target directory that does not exist on disk returns an error string."""
        path = self._make(tmp_path)
        result = _resolve_transform(
            {"path": path, "target_dir": str(tmp_path / "nonexistent")},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "Target folder not found" in result

    def test_non_string_target_dir_treated_as_absent(self, tmp_path):
        """A non-string target_dir is coerced to '' (no folder move)."""
        path = self._make(tmp_path, "keep.jpg")
        # No rename, non-string target_dir → no-op.
        result = _resolve_transform(
            {"path": path, "target_dir": True},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert result is None

    # ── destination collision ──────────────────────────────────────────────────

    def test_dst_already_exists_is_error(self, tmp_path):
        """A destination filename that already exists on disk returns an error string."""
        path = self._make(tmp_path, "a.jpg")
        (tmp_path / "b.jpg").write_bytes(b"x")  # collision target
        result = _resolve_transform(
            {"path": path, "new_name": "b.jpg"},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert isinstance(result, str)
        assert "already exists" in result

    # ── no-op ──────────────────────────────────────────────────────────────────

    def test_same_name_no_target_dir_is_noop(self, tmp_path):
        """Renaming to the same name in the same directory is a no-op (returns None)."""
        path = self._make(tmp_path, "img.jpg")
        result = _resolve_transform(
            {"path": path, "new_name": "img.jpg"},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert result is None

    def test_no_new_name_no_target_dir_is_noop(self, tmp_path):
        """No rename and no folder move is a no-op (returns None)."""
        path = self._make(tmp_path, "img.jpg")
        result = _resolve_transform(
            {"path": path},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert result is None

    # ── success cases ──────────────────────────────────────────────────────────

    def test_rename_only_returns_correct_dst(self, tmp_path):
        """A rename-only transform returns the correct (src, dst) path pair."""
        path = self._make(tmp_path, "old.jpg")
        src, dst = _resolve_transform(
            {"path": path, "new_name": "new.jpg"},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert src == (tmp_path / "old.jpg").resolve()
        assert dst == (tmp_path / "new.jpg").resolve()

    def test_move_only_returns_correct_dst(self, tmp_path):
        """A move-only transform returns the correct (src, dst) path pair."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        path = self._make(tmp_path, "img.jpg")
        src, dst = _resolve_transform(
            {"path": path, "target_dir": str(subdir)},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert src == (tmp_path / "img.jpg").resolve()
        assert dst == (subdir / "img.jpg").resolve()

    def test_rename_and_move_returns_correct_dst(self, tmp_path):
        """A combined rename+move returns the correctly named file in the target directory."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        path = self._make(tmp_path, "old.jpg")
        src, dst = _resolve_transform(
            {"path": path, "new_name": "new.jpg", "target_dir": str(subdir)},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert src == (tmp_path / "old.jpg").resolve()
        assert dst == (subdir / "new.jpg").resolve()

    def test_move_to_same_dir_with_same_name_is_noop(self, tmp_path):
        """target_dir = current dir + same name → dst == src → no-op."""
        path = self._make(tmp_path, "img.jpg")
        result = _resolve_transform(
            {"path": path, "target_dir": str(tmp_path)},
            keep_set={path},
            dir_resolved=tmp_path.resolve(),
        )
        assert result is None


# ── load_vcache / save_vcache ──────────────────────────────────────────────────

class TestVCacheIO:
    def test_missing_file_returns_empty(self, tmp_path):
        """A non-existent vcache file returns an empty dict rather than raising."""
        assert load_vcache(str(tmp_path / "nonexistent.csv")) == {}

    def test_round_trip_preserves_data(self, tmp_path):
        """Video cache data saved and reloaded with save/load_vcache is bit-for-bit identical."""
        cache_path = str(tmp_path / "vcache.csv")
        original = {
            "\\videos\\a.mp4": {
                "size": 1000, "mtime": 1708531200.0,
                "duration": 120.5, "vhash": ["f" * 16, "0" * 16],
            },
        }
        save_vcache(cache_path, original)
        loaded = load_vcache(cache_path)
        assert loaded == original

    def test_load_normalizes_types(self, tmp_path):
        """Loaded vcache entries have the correct Python types (int size, float mtime, list vhash)."""
        cache_path = str(tmp_path / "vcache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VCACHE_FIELDS)
            writer.writeheader()
            writer.writerow({
                "path": "\\videos\\a.mp4", "size": "1234",
                "mtime": "1708531200.5", "duration": "60.0",
                "vhash": "0" * 16,
            })
        entry = load_vcache(cache_path)["\\videos\\a.mp4"]
        assert isinstance(entry["size"], int)
        assert isinstance(entry["mtime"], float)
        assert isinstance(entry["duration"], float)
        assert isinstance(entry["vhash"], list)

    def test_invalid_vhash_row_skipped(self, tmp_path):
        """A row with a non-hex vhash value is skipped entirely on load."""
        cache_path = str(tmp_path / "vcache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VCACHE_FIELDS)
            writer.writeheader()
            writer.writerow({
                "path": "\\videos\\bad.mp4", "size": 100,
                "mtime": 1.0, "duration": 60.0, "vhash": "not_hex",
            })
        assert load_vcache(cache_path) == {}

    def test_empty_vhash_row_skipped(self, tmp_path):
        """A row with an empty vhash string is skipped on load."""
        cache_path = str(tmp_path / "vcache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VCACHE_FIELDS)
            writer.writeheader()
            writer.writerow({
                "path": "\\videos\\empty.mp4", "size": 100,
                "mtime": 1.0, "duration": 60.0, "vhash": "",
            })
        assert load_vcache(cache_path) == {}

    def test_corrupt_file_returns_empty_dict(self, tmp_path):
        """A vcache file with invalid content returns an empty dict rather than raising."""
        cache_path = str(tmp_path / "vcache.csv")
        Path(cache_path).write_text("garbage\ndata\n", encoding="utf-8")
        result = load_vcache(cache_path)
        assert isinstance(result, dict)

    def test_save_overwrites_existing_file(self, tmp_path):
        """save_vcache replaces an existing file rather than appending to it."""
        cache_path = str(tmp_path / "vcache.csv")
        save_vcache(cache_path, {})
        data = {
            "\\videos\\a.mp4": {
                "size": 1, "mtime": 1.0, "duration": 5.0, "vhash": ["0" * 16],
            },
        }
        save_vcache(cache_path, data)
        loaded = load_vcache(cache_path)
        assert "\\videos\\a.mp4" in loaded

    def test_multiple_frame_hashes_round_trip(self, tmp_path):
        """Multiple per-frame hash strings survive the save/load round-trip intact."""
        cache_path = str(tmp_path / "vcache.csv")
        frames = [format(i, "016x") for i in range(N_VIDEO_FRAMES)]
        original = {
            "\\videos\\a.mp4": {
                "size": 500, "mtime": 1.0, "duration": 30.0, "vhash": frames,
            },
        }
        save_vcache(cache_path, original)
        loaded = load_vcache(cache_path)
        assert loaded["\\videos\\a.mp4"]["vhash"] == frames


# ── _video_distance ────────────────────────────────────────────────────────────

class TestVideoDistance:
    def test_identical_frames_zero_distance(self):
        """Two identical frame-hash sequences have a distance of 0.0."""
        frames = [0xAABBCCDD00112233, 0x1122334455667788]
        assert _video_distance(frames, frames) == 0.0

    def test_all_bits_flipped_max_distance(self):
        """Completely opposite frame hashes produce the maximum distance of 64.0."""
        a = [0x0000000000000000]
        b = [0xFFFFFFFFFFFFFFFF]
        assert _video_distance(a, b) == 64.0

    def test_mean_across_frames(self):
        """The distance is the mean Hamming distance across all compared frame pairs."""
        # Frame 0: 0 bits differ; frame 1: 64 bits differ → mean = 32.0
        a = [0x0000000000000000, 0x0000000000000000]
        b = [0x0000000000000000, 0xFFFFFFFFFFFFFFFF]
        assert _video_distance(a, b) == 32.0

    def test_unequal_lengths_uses_minimum(self):
        """When frame lists differ in length, only the shorter list's frames are compared."""
        a = [0x0000000000000000, 0xFFFFFFFFFFFFFFFF]
        b = [0x0000000000000000]
        assert _video_distance(a, b) == 0.0

    def test_empty_frames_returns_inf(self):
        """Empty frame lists return infinity, ensuring they are never grouped as duplicates."""
        assert _video_distance([], []) == float("inf")

    def test_partial_bit_difference(self):
        """A partial bit difference is reflected exactly in the distance score."""
        # 0x000000000000000F has 4 set bits → distance 4 from 0x0
        a = [0x0000000000000000]
        b = [0x000000000000000F]
        assert _video_distance(a, b) == 4.0


# ── group_video_duplicates ─────────────────────────────────────────────────────

class TestGroupVideoDuplicates:
    """Uses synthetic vhashes dicts (no real video files needed)."""

    def _entry(self, tmp_path, name, duration, frames, size=1000):
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        return str(p), (duration, frames, size)

    def test_empty_input(self):
        """An empty vhash dict returns no duplicate groups."""
        assert group_video_duplicates({}, threshold=10) == []

    def test_single_video_not_grouped(self, tmp_path):
        """A single video cannot form a duplicate group."""
        path, entry = self._entry(tmp_path, "a.mp4", 60.0, [0] * 8)
        assert group_video_duplicates({path: entry}, threshold=10) == []

    def test_identical_hashes_grouped(self, tmp_path):
        """Two videos with identical frame hashes are placed in one duplicate group."""
        frames = [0] * 8
        pa, ea = self._entry(tmp_path, "a.mp4", 60.0, frames, size=2000)
        pb, eb = self._entry(tmp_path, "b.mp4", 60.0, frames, size=1000)
        groups = group_video_duplicates({pa: ea, pb: eb}, threshold=0)
        assert len(groups) == 1
        assert {p for p, _ in groups[0]} == {pa, pb}

    def test_distant_hashes_not_grouped(self, tmp_path):
        """Videos with maximum hash distance are not grouped as duplicates."""
        pa, ea = self._entry(tmp_path, "a.mp4", 60.0, [0x0000000000000000] * 8)
        pb, eb = self._entry(tmp_path, "b.mp4", 60.0, [0xFFFFFFFFFFFFFFFF] * 8)
        groups = group_video_duplicates({pa: ea, pb: eb}, threshold=10)
        assert groups == []

    def test_threshold_boundary(self, tmp_path):
        """Videos at or below the mean-distance threshold are grouped; above it they are not."""
        # One frame differs by 4 bits, rest identical → mean distance = 4/8 = 0.5
        frames_a = [0x0000000000000000] * 8
        frames_b = [0x000000000000000F] + [0x0000000000000000] * 7
        pa, ea = self._entry(tmp_path, "a.mp4", 60.0, frames_a)
        pb, eb = self._entry(tmp_path, "b.mp4", 60.0, frames_b)
        assert group_video_duplicates({pa: ea, pb: eb}, threshold=0) == []
        assert len(group_video_duplicates({pa: ea, pb: eb}, threshold=1)) == 1

    def test_duration_filter_excludes_far_durations(self, tmp_path):
        """Videos whose durations differ by more than the tolerance are not grouped."""
        frames = [0] * 8
        # 60s vs 200s: diff=140 > max(10, 0.05*200=10) → excluded
        pa, ea = self._entry(tmp_path, "a.mp4", 60.0, frames)
        pb, eb = self._entry(tmp_path, "b.mp4", 200.0, frames)
        assert group_video_duplicates({pa: ea, pb: eb}, threshold=0) == []

    def test_duration_filter_includes_within_relative_tolerance(self, tmp_path):
        """Videos within the relative duration tolerance are eligible for grouping."""
        frames = [0] * 8
        # 60s vs 67s: diff=7 < max(10, 0.05*67=3.35) = 10 → within tolerance
        pa, ea = self._entry(tmp_path, "a.mp4", 60.0, frames)
        pb, eb = self._entry(tmp_path, "b.mp4", 67.0, frames)
        assert len(group_video_duplicates({pa: ea, pb: eb}, threshold=0)) == 1

    def test_duration_absolute_tolerance_for_short_videos(self, tmp_path):
        """Short videos use an absolute (not percentage) duration tolerance."""
        # Short videos: 5s vs 14s, diff=9 < max(10, 0.05*14=0.7) = 10 → within
        frames = [0] * 8
        pa, ea = self._entry(tmp_path, "a.mp4", 5.0, frames)
        pb, eb = self._entry(tmp_path, "b.mp4", 14.0, frames)
        assert len(group_video_duplicates({pa: ea, pb: eb}, threshold=0)) == 1

    def test_duration_absolute_tolerance_boundary(self, tmp_path):
        """Videos just outside the absolute duration tolerance boundary are excluded."""
        # 5s vs 16s: diff=11 > max(10, 0.05*16=0.8) = 10 → excluded
        frames = [0] * 8
        pa, ea = self._entry(tmp_path, "a.mp4", 5.0, frames)
        pb, eb = self._entry(tmp_path, "b.mp4", 16.0, frames)
        assert group_video_duplicates({pa: ea, pb: eb}, threshold=0) == []

    def test_within_group_sorted_largest_first(self, tmp_path):
        """Within a video group, the largest file (by byte size) appears first."""
        frames = [0] * 8
        pa, ea = self._entry(tmp_path, "large.mp4", 60.0, frames, size=5000)
        pb, eb = self._entry(tmp_path, "small.mp4", 60.0, frames, size=1000)
        groups = group_video_duplicates({pa: ea, pb: eb}, threshold=0)
        assert len(groups) == 1
        assert groups[0][0][0] == pa
        assert groups[0][0][1] == 5000

    def test_groups_sorted_most_files_first(self, tmp_path):
        """Video groups with more files appear before groups with fewer files."""
        frames_a = [0x0000000000000000] * 8
        frames_b = [0xFFFFFFFFFFFFFFFF] * 8
        trio = [self._entry(tmp_path, f"trio{i}.mp4", 60.0, frames_a) for i in range(3)]
        pair = [self._entry(tmp_path, f"pair{i}.mp4", 300.0, frames_b) for i in range(2)]
        vhashes = dict(trio + pair)
        groups = group_video_duplicates(vhashes, threshold=0)
        assert len(groups) == 2
        assert len(groups[0]) == 3

    def test_transitive_grouping_via_union_find(self, tmp_path):
        """A-B and B-C connected → all three in one group."""
        frames_a = [0x0000000000000000] * 8
        frames_b = [0x000000000000000F] + [0x0000000000000000] * 7  # distance 0.5 from a
        pa, ea = self._entry(tmp_path, "a.mp4", 60.0, frames_a)
        pb, eb = self._entry(tmp_path, "b.mp4", 60.0, frames_b)
        pc, ec = self._entry(tmp_path, "c.mp4", 60.0, frames_a)
        groups = group_video_duplicates({pa: ea, pb: eb, pc: ec}, threshold=1)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_no_file_appears_in_two_groups(self, tmp_path):
        """Each video appears in at most one duplicate group."""
        frames_a = [0x0000000000000000] * 8
        frames_b = [0xFFFFFFFFFFFFFFFF] * 8
        group1 = [self._entry(tmp_path, f"g1_{i}.mp4", 60.0, frames_a) for i in range(3)]
        group2 = [self._entry(tmp_path, f"g2_{i}.mp4", 300.0, frames_b) for i in range(2)]
        vhashes = dict(group1 + group2)
        groups = group_video_duplicates(vhashes, threshold=0)
        all_paths = [p for g in groups for p, _ in g]
        assert len(all_paths) == len(set(all_paths))


# ── ffmpeg_available ───────────────────────────────────────────────────────────

class TestFfmpegAvailable:
    def test_returns_true_when_ffmpeg_found(self):
        """Returns True when ffmpeg is found in PATH."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert ffmpeg_available() is True

    def test_returns_false_when_ffmpeg_missing(self):
        """Returns False when ffmpeg is absent from PATH."""
        with patch("shutil.which", return_value=None):
            assert ffmpeg_available() is False


# ── _get_video_duration ────────────────────────────────────────────────────────

class TestGetVideoDuration:
    def _run_ok(self, stdout="120.5\n"):
        r = MagicMock()
        r.returncode = 0
        r.stdout = stdout
        r.stderr = ""
        return r

    def _run_fail(self, stderr=""):
        r = MagicMock()
        r.returncode = 1
        r.stdout = ""
        r.stderr = stderr
        return r

    def test_uses_ffprobe_when_available(self, tmp_path):
        """Uses ffprobe to obtain the duration when it is available in PATH."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("subprocess.run", return_value=self._run_ok("90.0\n")) as mock_run:
            dur = _get_video_duration(str(dummy))
        assert dur == 90.0
        assert mock_run.call_args[0][0][0] == "ffprobe"

    def test_falls_back_to_ffmpeg_when_ffprobe_absent(self, tmp_path):
        """Falls back to parsing ffmpeg stderr 'Duration:' line when ffprobe is absent."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        stderr = "... Duration: 00:02:03.50, start: ..."
        with patch("shutil.which", return_value=None), \
             patch("subprocess.run", return_value=self._run_fail(stderr=stderr)):
            dur = _get_video_duration(str(dummy))
        assert abs(dur - (2 * 60 + 3.5)) < 0.01

    def test_raises_when_neither_works(self, tmp_path):
        """Raises RuntimeError when neither ffprobe nor ffmpeg can provide a duration."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("shutil.which", return_value=None), \
             patch("subprocess.run", return_value=self._run_fail(stderr="no duration here")):
            with pytest.raises(RuntimeError):
                _get_video_duration(str(dummy))


# ── compute_video_hashes ───────────────────────────────────────────────────────

class TestComputeVideoHashes:
    # compute_video_hashes outputs raw RGB24 at FRAME_W×FRAME_H (64×64).
    # Each frame is 64*64*3 = 12 288 bytes.
    _FRAME_W, _FRAME_H = 64, 64

    def _raw_frame(self, flip=False) -> bytes:
        """Return one raw RGB24 64×64 frame with a white/black split.

        flip=False: white on bottom half; flip=True: white on top half.
        The spatial structure guarantees distinct low-frequency DCT content so
        pHash produces different values for the two orientations.
        """
        img = Image.new("RGB", (self._FRAME_W, self._FRAME_H), 0)
        half = Image.new("RGB", (self._FRAME_W, self._FRAME_H // 2), (255, 255, 255))
        img.paste(half, (0, 0) if flip else (0, self._FRAME_H // 2))
        return img.tobytes()  # raw RGB24, exactly 12 288 bytes

    def _run_ok(self, flip=False, n=8):
        """Return a mock subprocess result with *n* raw RGB24 frames."""
        r = MagicMock()
        r.returncode = 0
        r.stdout = self._raw_frame(flip) * n
        return r

    def _run_fail(self):
        r = MagicMock()
        r.returncode = 1
        r.stdout = b""
        return r

    def test_returns_duration_and_n_hashes(self, tmp_path):
        """Returns a (duration, frames) tuple with exactly n integer frame hashes."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("idem._get_video_duration", return_value=80.0), \
             patch("subprocess.run", return_value=self._run_ok()):
            duration, frames = compute_video_hashes(str(dummy), n=4)
        assert duration == 80.0
        assert len(frames) == 4
        assert all(isinstance(f, int) for f in frames)

    def test_n_controls_frame_count(self, tmp_path):
        """The n parameter controls exactly how many frame hashes are returned."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("idem._get_video_duration", return_value=60.0), \
             patch("subprocess.run", return_value=self._run_ok()):
            _, frames = compute_video_hashes(str(dummy), n=3)
        assert len(frames) == 3

    def test_ffmpeg_failure_raises(self, tmp_path):
        """An ffmpeg subprocess failure raises a RuntimeError."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("idem._get_video_duration", return_value=60.0), \
             patch("subprocess.run", return_value=self._run_fail()):
            with pytest.raises(RuntimeError):
                compute_video_hashes(str(dummy), n=4)

    def test_zero_duration_raises(self, tmp_path):
        """A zero-duration video raises a RuntimeError (cannot sample frames)."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("idem._get_video_duration", return_value=0.0):
            with pytest.raises(RuntimeError):
                compute_video_hashes(str(dummy), n=4)

    def test_identical_frames_produce_identical_hashes(self, tmp_path):
        """The same visual content always produces the same frame hash sequence."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("idem._get_video_duration", return_value=60.0), \
             patch("subprocess.run", return_value=self._run_ok()):
            _, frames1 = compute_video_hashes(str(dummy), n=4)
        with patch("idem._get_video_duration", return_value=60.0), \
             patch("subprocess.run", return_value=self._run_ok()):
            _, frames2 = compute_video_hashes(str(dummy), n=4)
        assert frames1 == frames2

    def test_different_frames_produce_different_hashes(self, tmp_path):
        """Different visual content (spatially flipped frames) produces different hashes."""
        # flip=False: white on bottom half; flip=True: white on top half
        # These have inverted spatial structure → different low-frequency DCT → different pHash
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")
        with patch("idem._get_video_duration", return_value=60.0), \
             patch("subprocess.run", return_value=self._run_ok(flip=False)):
            _, frames_a = compute_video_hashes(str(dummy), n=1)
        with patch("idem._get_video_duration", return_value=60.0), \
             patch("subprocess.run", return_value=self._run_ok(flip=True)):
            _, frames_b = compute_video_hashes(str(dummy), n=1)
        assert frames_a != frames_b


# ── build_video_hashes ─────────────────────────────────────────────────────────

class TestBuildVideoHashes:
    FAKE_DURATION = 60.0
    FAKE_FRAMES = [0xAABBCCDD00112233] * N_VIDEO_FRAMES

    def _files(self, tmp_path, *names, size=100):
        paths = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"x" * size)
            paths.append(str(p))
        return paths

    def _cached_entry(self, path):
        st = os.stat(path)
        return {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "duration": self.FAKE_DURATION,
            "vhash": [format(h, "016x") for h in self.FAKE_FRAMES],
        }

    def test_new_file_hashed_and_counted(self, tmp_path):
        """A video file not in the vcache is hashed and counted as new."""
        (path,) = self._files(tmp_path, "a.mp4")
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            vhashes, new_c, rehashed_c, errors = build_video_hashes([path], {})
        assert new_c == 1
        assert rehashed_c == 0
        assert errors == 0
        assert path in vhashes

    def test_vhashes_tuple_structure(self, tmp_path):
        """The vhashes dict stores (duration, frames, size) tuples keyed by file path."""
        (path,) = self._files(tmp_path, "a.mp4", size=500)
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            vhashes, _, _, _ = build_video_hashes([path], {})
        dur, frames, size = vhashes[path]
        assert dur == self.FAKE_DURATION
        assert frames == self.FAKE_FRAMES
        assert size == 500

    def test_cache_hit_skips_recompute(self, tmp_path):
        """A video with unchanged size/mtime is served from cache without calling compute_video_hashes."""
        (path,) = self._files(tmp_path, "a.mp4")
        cache = {path_without_drive(path): self._cached_entry(path)}
        with patch("idem.compute_video_hashes") as mock_compute:
            _, new_c, rehashed_c, errors = build_video_hashes([path], cache)
            mock_compute.assert_not_called()
        assert new_c == 0
        assert rehashed_c == 0
        assert errors == 0

    def test_stale_size_triggers_recompute(self, tmp_path):
        """A video whose cached size no longer matches disk is re-hashed and counted as re-hashed."""
        (path,) = self._files(tmp_path, "a.mp4")
        entry = self._cached_entry(path)
        entry["size"] += 1
        cache = {path_without_drive(path): entry}
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            _, _, rehashed_c, errors = build_video_hashes([path], cache)
        assert rehashed_c == 1
        assert errors == 0

    def test_stale_mtime_triggers_recompute(self, tmp_path):
        """A video whose cached mtime differs beyond the tolerance is re-hashed."""
        (path,) = self._files(tmp_path, "a.mp4")
        entry = self._cached_entry(path)
        entry["mtime"] -= _TS_TOLERANCE + 10
        cache = {path_without_drive(path): entry}
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            _, _, rehashed_c, errors = build_video_hashes([path], cache)
        assert rehashed_c == 1

    def test_compute_error_counted_not_in_vhashes(self, tmp_path):
        """Videos that fail to hash are counted as errors and excluded from vhashes."""
        (path,) = self._files(tmp_path, "bad.mp4")
        with patch("idem.compute_video_hashes",
                   side_effect=RuntimeError("ffmpeg failed")):
            vhashes, _, _, errors = build_video_hashes([path], {})
        assert errors == 1
        assert path not in vhashes

    def test_cache_key_is_driveless(self, tmp_path):
        """Video cache keys are stored without drive letters for cross-platform consistency."""
        (path,) = self._files(tmp_path, "a.mp4")
        cache = {}
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            build_video_hashes([path], cache)
        for key in cache:
            assert not (len(key) >= 2 and key[1] == ":"), f"Drive letter in key: {key!r}"

    def test_cache_stores_hex_frame_strings(self, tmp_path):
        """Frame hashes are stored in the vcache as hex strings, not raw integers."""
        (path,) = self._files(tmp_path, "a.mp4")
        cache = {}
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            build_video_hashes([path], cache)
        key = path_without_drive(path)
        assert isinstance(cache[key]["vhash"], list)
        assert all(isinstance(h, str) for h in cache[key]["vhash"])

    def test_multiple_files(self, tmp_path):
        """Multiple video files are all hashed in a single call, each counted independently."""
        paths = self._files(tmp_path, "a.mp4", "b.mp4", "c.mp4")
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            vhashes, new_c, _, errors = build_video_hashes(paths, {})
        assert len(vhashes) == 3
        assert new_c == 3
        assert errors == 0

    def test_cache_appended_via_writer(self, tmp_path):
        """Newly computed video hashes are flushed to the vcache file incrementally via the writer."""
        (path,) = self._files(tmp_path, "a.mp4")
        cache_path = str(tmp_path / VCACHE_FILENAME)
        with patch("idem.compute_video_hashes",
                   return_value=(self.FAKE_DURATION, self.FAKE_FRAMES)):
            cache_out = _open_vcache_for_append(cache_path)
            try:
                build_video_hashes([path], {}, cache_out)
            finally:
                cache_out.close()
        loaded = load_vcache(cache_path)
        assert path_without_drive(path) in loaded


# ── Priority 4 — Helper unit tests ────────────────────────────────────────────


class TestValidHex:
    def test_empty_string_returns_false(self):
        """An empty string is not a valid hex string."""
        assert _valid_hex("") is False

    def test_single_digit(self):
        """A single decimal digit is a valid hex character."""
        assert _valid_hex("0") is True

    def test_all_lowercase_hex(self):
        """All-lowercase hex characters (a-f, 0-9) are accepted."""
        assert _valid_hex("deadbeef") is True

    def test_all_uppercase_hex(self):
        """All-uppercase hex characters (A-F, 0-9) are accepted."""
        assert _valid_hex("DEADBEEF") is True

    def test_mixed_case_hex(self):
        """Mixed-case hex characters are accepted."""
        assert _valid_hex("aAbBcCdD") is True

    def test_invalid_char(self):
        """Non-hex characters (e.g. 'z') cause the function to return False."""
        assert _valid_hex("zzzz") is False

    def test_hex_prefix_rejected(self):
        """The '0x' prefix is rejected because 'x' is not a valid hex character."""
        # 'x' is not a valid hex character
        assert _valid_hex("0x1234") is False


class TestFmtSizeExtra:
    def test_terabytes(self):
        """Values in the TB range are formatted with a 'TB' suffix."""
        assert "TB" in fmt_size(5 * 1024 ** 4)

    def test_boundary_1023_is_bytes(self):
        """1023 bytes is still formatted as bytes, not KB (boundary just below 1 KB)."""
        assert "B" in fmt_size(1023)

    def test_boundary_1024_is_kb(self):
        """1024 bytes is formatted as 1 KB (exact boundary)."""
        assert "KB" in fmt_size(1024)


class TestPathWithoutDriveExtra:
    def test_unc_path(self):
        """UNC paths (\\\\server\\share\\...) have the server/share prefix stripped."""
        result = path_without_drive("\\\\server\\share\\file.jpg")
        assert "file.jpg" in result
        assert not result.startswith("\\\\server")


# ── Priority 1 — Zero-coverage core functions ─────────────────────────────────


class TestParseSize:
    def test_bare_integer(self):
        """A bare integer string (no suffix) is parsed as a byte count."""
        assert parse_size("500") == 500

    def test_b_suffix(self):
        """A 'b' suffix is treated as bytes."""
        assert parse_size("512b") == 512

    def test_kb_lowercase(self):
        """Lowercase 'kb' suffix is parsed as kilobytes."""
        assert parse_size("50kb") == 50 * 1024

    def test_kb_uppercase(self):
        """Uppercase 'KB' suffix is also parsed as kilobytes."""
        assert parse_size("50KB") == 50 * 1024

    def test_mb(self):
        """'MB' suffix is parsed as megabytes."""
        assert parse_size("2MB") == 2 * 1024 ** 2

    def test_gb(self):
        """'GB' suffix is parsed as gigabytes."""
        assert parse_size("1GB") == 1024 ** 3

    def test_tb(self):
        """'TB' suffix is parsed as terabytes."""
        assert parse_size("1TB") == 1024 ** 4

    def test_decimal_kb(self):
        """Fractional kilobyte values are parsed and truncated to an integer."""
        assert parse_size("1.5kb") == int(1.5 * 1024)

    def test_decimal_mb(self):
        """Fractional megabyte values are parsed and truncated to an integer."""
        assert parse_size("2.5MB") == int(2.5 * 1024 ** 2)

    def test_case_insensitive_mb(self):
        """Suffix matching is case-insensitive ('mb' same as 'MB')."""
        assert parse_size("2mb") == 2 * 1024 ** 2

    def test_leading_whitespace_stripped(self):
        """Leading whitespace is stripped before parsing."""
        assert parse_size("  10kb") == 10 * 1024

    def test_invalid_string_raises(self):
        """A string with no recognisable size suffix raises an exception."""
        with pytest.raises((ValueError, TypeError)):
            parse_size("abc")

    def test_empty_string_raises(self):
        """An empty string raises an exception."""
        with pytest.raises((ValueError, TypeError)):
            parse_size("")


class TestFileChecksum:
    def test_result_is_64_char_hex(self, tmp_path):
        """The checksum is a 64-character hex string (SHA-256 digest)."""
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello world")
        result = _file_checksum(str(p))
        assert len(result) == 64
        assert _valid_hex(result)

    def test_identical_content_same_checksum(self, tmp_path):
        """Identical file content always produces the same checksum."""
        data = b"same content here" * 10
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(data)
        p2.write_bytes(data)
        assert _file_checksum(str(p1)) == _file_checksum(str(p2))

    def test_different_content_different_checksum(self, tmp_path):
        """Different file content produces different checksums."""
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"content A")
        p2.write_bytes(b"content B")
        assert _file_checksum(str(p1)) != _file_checksum(str(p2))

    def test_same_bytes_different_size_different_checksum(self, tmp_path):
        """Size is mixed into the digest first, so two files with different sizes
        produce different checksums even if their sampled windows are identical."""
        old_threshold = idem_module._FULL_HASH_THRESHOLD
        old_sample = idem_module._SAMPLE_SIZE
        try:
            idem_module._FULL_HASH_THRESHOLD = 30
            idem_module._SAMPLE_SIZE = 10
            p30 = tmp_path / "thirty.bin"
            p31 = tmp_path / "thirtyone.bin"
            # Both files start with the same 10 bytes; they differ only in total size.
            p30.write_bytes(b"0123456789" * 3)          # 30 bytes
            p31.write_bytes(b"0123456789" * 3 + b"X")   # 31 bytes
            assert _file_checksum(str(p30)) != _file_checksum(str(p31))
        finally:
            idem_module._FULL_HASH_THRESHOLD = old_threshold
            idem_module._SAMPLE_SIZE = old_sample

    def test_large_file_sampling_misses_gap_content(self, tmp_path):
        """Two files that differ only in un-sampled gap bytes produce the same
        checksum, confirming the sampling path is exercised."""
        old_threshold = idem_module._FULL_HASH_THRESHOLD
        old_sample = idem_module._SAMPLE_SIZE
        try:
            idem_module._FULL_HASH_THRESHOLD = 30
            idem_module._SAMPLE_SIZE = 10
            # For a 40-byte file with sample_size=10:
            #   window 1: bytes [0:10]
            #   window 2: seek(40//2 - 10//2 = 15) → bytes [15:25]
            #   window 3: seek(-10, 2) → bytes [30:40]
            # Un-sampled regions: [10:15] and [25:30]
            prefix = b"P" * 10
            gap    = b"G" * 5
            mid    = b"M" * 10
            gap2   = b"G" * 5
            suffix = b"S" * 10
            file_a = prefix + gap  + mid + gap2 + suffix   # 40 bytes, gap=G
            file_b = prefix + b"X"*5 + mid + b"X"*5 + suffix  # 40 bytes, gap=X
            p_a = tmp_path / "file_a.bin"
            p_b = tmp_path / "file_b.bin"
            p_a.write_bytes(file_a)
            p_b.write_bytes(file_b)
            assert _file_checksum(str(p_a)) == _file_checksum(str(p_b))
        finally:
            idem_module._FULL_HASH_THRESHOLD = old_threshold
            idem_module._SAMPLE_SIZE = old_sample


class TestCollectStale:
    def _db_entry(self, path):
        st = os.stat(path)
        return {"size": st.st_size, "mtime": st.st_mtime}

    def test_new_file_not_in_db_is_returned(self, tmp_path):
        """A file not yet in the DB is added to the compute list."""
        p = tmp_path / "new.bin"
        p.write_bytes(b"data")
        result = _collect_stale([str(p)], {})
        assert str(p) in result

    def test_unchanged_file_is_cache_hit(self, tmp_path):
        """A file with matching size and mtime is a cache hit and not added to compute list."""
        p = tmp_path / "cached.bin"
        p.write_bytes(b"data")
        db = {path_without_drive(str(p)): self._db_entry(str(p))}
        result = _collect_stale([str(p)], db)
        assert result == []

    def test_size_change_makes_stale(self, tmp_path):
        """A change in file size marks the cache entry as stale and removes it from the DB."""
        p = tmp_path / "file.bin"
        p.write_bytes(b"short")
        k = path_without_drive(str(p))
        db = {k: {"size": 999, "mtime": os.stat(str(p)).st_mtime}}
        result = _collect_stale([str(p)], db)
        assert str(p) in result
        assert k not in db  # stale entry removed

    def test_mtime_over_tolerance_makes_stale(self, tmp_path):
        """A mtime difference beyond the tolerance marks the entry as stale."""
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        actual_mtime = os.stat(str(p)).st_mtime
        k = path_without_drive(str(p))
        db = {k: {"size": p.stat().st_size, "mtime": actual_mtime + _TS_TOLERANCE + 1}}
        result = _collect_stale([str(p)], db)
        assert str(p) in result

    def test_mtime_within_tolerance_is_hit(self, tmp_path):
        """A mtime difference within the tolerance is still treated as a cache hit."""
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        actual_mtime = os.stat(str(p)).st_mtime
        k = path_without_drive(str(p))
        db = {k: {"size": p.stat().st_size, "mtime": actual_mtime + _TS_TOLERANCE / 2}}
        result = _collect_stale([str(p)], db)
        assert result == []

    def test_mtime_at_boundary_is_stale(self, tmp_path):
        """A mtime difference exactly at the tolerance boundary is stale (strict < check)."""
        # Check is `< _TS_TOLERANCE` (strict), so exactly at boundary → stale
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        actual_mtime = os.stat(str(p)).st_mtime
        k = path_without_drive(str(p))
        db = {k: {"size": p.stat().st_size, "mtime": actual_mtime + _TS_TOLERANCE}}
        result = _collect_stale([str(p)], db)
        assert str(p) in result

    def test_oserror_on_stat_adds_to_compute_preserves_db_entry(self, tmp_path):
        """An OSError during stat adds the file to compute but preserves the existing DB entry."""
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        k = path_without_drive(str(p))
        db = {k: {"size": p.stat().st_size, "mtime": os.stat(str(p)).st_mtime}}
        with patch("idem_module.os.stat" if False else "idem.os.stat", side_effect=OSError("no access")):
            result = _collect_stale([str(p)], db)
        assert str(p) in result
        assert k in db  # entry NOT deleted on OSError


class TestGroupExactDuplicates:
    def _entry(self, checksum, size):
        return {"checksum": checksum, "size": size, "path": "unused",
                "mtime": 0.0, "ctime": 0.0}

    def test_empty_db_returns_empty(self):
        """An empty DB returns no duplicate groups."""
        assert group_exact_duplicates({}, {}) == []

    def test_unique_checksums_no_group(self):
        """Files with all-different checksums form no duplicate groups."""
        db = {
            "\\a.jpg": self._entry("aaa", 100),
            "\\b.jpg": self._entry("bbb", 200),
        }
        assert group_exact_duplicates(db, {}) == []

    def test_two_files_same_checksum_form_one_group(self):
        """Two files with the same checksum form exactly one duplicate group."""
        db = {
            "\\a.jpg": self._entry("abc123", 100),
            "\\b.jpg": self._entry("abc123", 200),
        }
        groups = group_exact_duplicates(db, {})
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_largest_first_within_group(self):
        """Within an exact-duplicate group, the file with the larger size appears first."""
        db = {
            "\\small.jpg": self._entry("same", 100),
            "\\large.jpg": self._entry("same", 500),
        }
        groups = group_exact_duplicates(db, {})
        assert len(groups) == 1
        assert groups[0][0][1] == 500   # largest first
        assert groups[0][1][1] == 100

    def test_multiple_groups_sorted_largest_group_first(self):
        """Groups are sorted so the group with the most members appears first."""
        db = {
            "\\a1.jpg": self._entry("hash_a", 100),
            "\\a2.jpg": self._entry("hash_a", 100),
            "\\a3.jpg": self._entry("hash_a", 100),
            "\\b1.jpg": self._entry("hash_b", 200),
            "\\b2.jpg": self._entry("hash_b", 200),
        }
        groups = group_exact_duplicates(db, {})
        assert len(groups) == 2
        assert len(groups[0]) == 3   # group of 3 first

    def test_abs_path_map_fallback(self):
        """Keys not present in abs_path_map fall back to using the key itself as the path."""
        db = {
            "\\key_a.jpg": self._entry("dup", 100),
            "\\key_b.jpg": self._entry("dup", 100),
        }
        # abs_path_map only maps one key; the other falls back to the key itself
        abs_path_map = {"\\key_a.jpg": "/resolved/key_a.jpg"}
        groups = group_exact_duplicates(db, abs_path_map)
        assert len(groups) == 1
        paths = {p for p, _ in groups[0]}
        assert "/resolved/key_a.jpg" in paths
        assert "\\key_b.jpg" in paths


class TestBuildExactIndex:
    def test_new_files_hashed_and_counted(self, tmp_path):
        """New files are checksummed and new_count reflects the number processed."""
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"file a content")
        p2.write_bytes(b"file b content")
        db_path = str(tmp_path / "db.csv")
        db, new_count, gone_count, errors = build_exact_index(
            [str(p1), str(p2)], db_path
        )
        assert new_count == 2
        assert gone_count == 0
        assert errors == 0
        assert len(db) == 2

    def test_gone_files_removed(self, tmp_path):
        """Files no longer on disk are removed from the index and counted in gone_count."""
        p1 = tmp_path / "keep.bin"
        p2 = tmp_path / "gone.bin"
        p1.write_bytes(b"keep")
        p2.write_bytes(b"gone")
        db_path = str(tmp_path / "db.csv")
        build_exact_index([str(p1), str(p2)], db_path)
        # Second call without p2
        db, new_count, gone_count, errors = build_exact_index([str(p1)], db_path)
        assert gone_count == 1
        assert len(db) == 1

    def test_cache_hit_no_recount(self, tmp_path):
        """A file unchanged between runs is served from cache (new_count stays 0)."""
        p = tmp_path / "file.bin"
        p.write_bytes(b"stable content")
        db_path = str(tmp_path / "db.csv")
        build_exact_index([str(p)], db_path)
        _, new_count, _, _ = build_exact_index([str(p)], db_path)
        assert new_count == 0

    def test_error_file_counted(self, tmp_path):
        """Files that cannot be checksummed (e.g. missing) are counted as errors."""
        p_real = tmp_path / "real.bin"
        p_real.write_bytes(b"real content")
        p_missing = str(tmp_path / "nonexistent.bin")
        db_path = str(tmp_path / "db.csv")
        db, new_count, _, errors = build_exact_index(
            [str(p_real), p_missing], db_path
        )
        assert errors == 1
        assert path_without_drive(str(p_real)) in db

    def test_return_tuple_structure(self, tmp_path):
        """The return value is a 4-tuple of (db, new_count, gone_count, errors)."""
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        db_path = str(tmp_path / "db.csv")
        result = build_exact_index([str(p)], db_path)
        assert isinstance(result, tuple)
        assert len(result) == 4
        db, new_count, gone_count, errors = result
        assert isinstance(db, dict)
        assert isinstance(new_count, int)
        assert isinstance(gone_count, int)
        assert isinstance(errors, int)


# ── Priority 2 — Correctness gaps ─────────────────────────────────────────────


class TestGroupDuplicatesDhash:
    def test_dhash_filter_prevents_false_positive_grouping(self, tmp_path):
        """Identical phash but maximum dhash distance should NOT be grouped at threshold=0."""
        p1 = str(tmp_path / "a.jpg")
        p2 = str(tmp_path / "b.jpg")
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        phash_int = 0                         # "0" * 16
        dhash_a   = 0                         # all zero bits
        dhash_b   = int("f" * 16, 16)         # all one bits, distance=64 from dhash_a
        hashes = {p1: (phash_int, dhash_a, 100), p2: (phash_int, dhash_b, 100)}
        assert group_duplicates(hashes, threshold=0) == []


class TestCacheIOExtra:
    def test_load_mixed_valid_and_corrupt_rows(self, tmp_path):
        """Valid rows are loaded successfully even when adjacent rows are corrupt."""
        cache_path = str(tmp_path / "cache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
            writer.writeheader()
            writer.writerow({"path": "\\valid.jpg", "size": "1000",
                             "mtime": "1.0", "phash": "0" * 16, "dhash": ""})
            writer.writerow({"path": "\\bad.jpg", "size": "not_an_int",
                             "mtime": "1.0", "phash": "0" * 16, "dhash": ""})
        result = load_cache(cache_path)
        assert len(result) == 1
        assert "\\valid.jpg" in result

    def test_load_missing_dhash_column_treated_as_empty(self, tmp_path):
        """A cache file without a dhash column is loaded with dhash defaulting to empty string."""
        cache_path = str(tmp_path / "cache.csv")
        fields_no_dhash = ["path", "size", "mtime", "phash"]
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields_no_dhash)
            writer.writeheader()
            writer.writerow({"path": "\\photo.jpg", "size": "500",
                             "mtime": "1.0", "phash": "0" * 16})
        result = load_cache(cache_path)
        assert "\\photo.jpg" in result
        assert result["\\photo.jpg"]["dhash"] == ""

    def test_save_mkstemp_failure_logs_warning(self, tmp_path, capsys):
        """An OS error during temp-file creation logs a warning but does not raise."""
        cache_path = str(tmp_path / "cache.csv")
        with patch("tempfile.mkstemp", side_effect=OSError("no space")):
            save_cache(cache_path, {})  # must not raise
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_save_replace_failure_cleans_up_temp(self, tmp_path, capsys):
        """An OS error during atomic replace logs a warning and cleans up the temp file."""
        cache_path = str(tmp_path / "cache.csv")
        with patch("os.replace", side_effect=OSError("replace failed")):
            save_cache(cache_path, {})  # must not raise
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        # No .tmp files should remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


class TestBuildHashesExtra:
    def test_mtime_within_tolerance_is_cache_hit(self, tmp_path):
        """A mtime shift within the tolerance window is treated as a cache hit."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        build_hashes([str(p)], cache)
        # Shift mtime by less than tolerance → should still be a cache hit
        key = path_without_drive(str(p))
        cache[key]["mtime"] += _TS_TOLERANCE / 2
        _, new_count, rehashed_count, _ = build_hashes([str(p)], cache)
        assert new_count == 0
        assert rehashed_count == 0

    def test_mtime_just_over_tolerance_triggers_rehash(self, tmp_path):
        """A mtime shift just over the tolerance triggers a re-hash."""
        p = write_image(tmp_path / "photo.jpg")
        cache = {}
        build_hashes([str(p)], cache)
        key = path_without_drive(str(p))
        cache[key]["mtime"] += _TS_TOLERANCE + 1
        _, new_count, rehashed_count, _ = build_hashes([str(p)], cache)
        assert rehashed_count == 1

    def test_stat_failure_during_scan_counts_as_error(self, tmp_path):
        """An OSError during os.stat is counted as an error and the file is skipped."""
        p = write_image(tmp_path / "photo.jpg")
        with patch("idem.os.stat", side_effect=OSError("permission denied")):
            hashes, _, _, error_count = build_hashes([str(p)], {})
        assert error_count == 1
        assert hashes == {}

    def test_mixed_errors_and_successes(self, tmp_path):
        """Successful hashes and errors are counted independently in the same run."""
        p1 = write_image(tmp_path / "ok1.jpg", color="red")
        p2 = write_image(tmp_path / "ok2.jpg", color="blue")
        p3 = tmp_path / "junk.jpg"
        p3.write_bytes(b"not an image at all, just junk bytes 0123456789")
        hashes, _, _, error_count = build_hashes([str(p1), str(p2), str(p3)], {})
        assert len(hashes) == 2
        assert error_count == 1


# ── Priority 3 — Robustness ────────────────────────────────────────────────────


class TestVCacheIOExtra:
    def test_partially_invalid_vhash_hex_row_skipped(self, tmp_path):
        """A vhash with one non-hex segment causes the whole row to be skipped."""
        cache_path = str(tmp_path / "vcache.csv")
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VCACHE_FIELDS)
            writer.writeheader()
            writer.writerow({
                "path": "\\videos\\bad.mp4", "size": 100,
                "mtime": 1.0, "duration": 60.0,
                "vhash": "0000000000000000,INVALID,0000000000000000",
            })
        assert load_vcache(cache_path) == {}


class TestGetVideoDurationExtra:
    def test_ffprobe_malformed_output_falls_back_to_ffmpeg(self, tmp_path):
        """If ffprobe returns non-float stdout, falls through to ffmpeg Duration parsing."""
        dummy = tmp_path / "v.mp4"
        dummy.write_bytes(b"x")

        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = "not_a_float\n"
        ffprobe_result.stderr = ""

        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 1
        ffmpeg_result.stdout = ""
        ffmpeg_result.stderr = "... Duration: 00:01:30.00, start: ..."

        call_count = {"n": 0}

        def fake_run(cmd, *a, **kw):
            call_count["n"] += 1
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return ffmpeg_result

        with patch("shutil.which", return_value="/usr/bin/ffprobe"), \
             patch("subprocess.run", side_effect=fake_run):
            dur = _get_video_duration(str(dummy))

        assert abs(dur - 90.0) < 0.01


# ── Priority 5 — Flask endpoint security ──────────────────────────────────────


flask = pytest.importorskip("flask")


@pytest.fixture
def review_app(tmp_path):
    from idem import _build_review_app
    img = write_image(tmp_path / "photo.jpg")
    groups = [[(str(img), img.stat().st_size)]]
    app, _ = _build_review_app(groups, str(tmp_path))
    app.config["TESTING"] = True
    return app, tmp_path


class TestReviewUI:
    def test_image_endpoint_serves_valid_file(self, review_app):
        """A valid image path served via /image returns HTTP 200."""
        app, tmp_path = review_app
        img_path = str(tmp_path / "photo.jpg")
        with app.test_client() as client:
            resp = client.get(f"/image?path={img_path}")
        assert resp.status_code == 200

    def test_image_endpoint_path_traversal_returns_403(self, review_app):
        """Path-traversal attempts to /image are rejected with HTTP 403."""
        app, tmp_path = review_app
        with app.test_client() as client:
            resp = client.get("/image?path=../../../etc/passwd")
        assert resp.status_code == 403

    def test_image_endpoint_missing_file_returns_404(self, review_app):
        """A valid but non-existent image path returns HTTP 404."""
        app, tmp_path = review_app
        missing = str(tmp_path / "deleted.jpg")
        with app.test_client() as client:
            resp = client.get(f"/image?path={missing}")
        assert resp.status_code == 404

    def test_thumbnail_endpoint_path_traversal_returns_403(self, review_app):
        """Path-traversal attempts to /thumbnail are rejected with HTTP 403."""
        app, tmp_path = review_app
        with app.test_client() as client:
            resp = client.get("/thumbnail?path=../../../etc/passwd")
        assert resp.status_code == 403

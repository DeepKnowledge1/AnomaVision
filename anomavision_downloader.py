"""
AnomaVision – Asset Downloader
==============================
Downloads bottle-class images and model weights into a global cache
(~/.anomavision/) on first use.

Download sources:
  - Models  : GitHub Release (assets-stable tag)
  - Images  : Google Drive   (large zip, no size limits)

Global cache layout
-------------------
    ~/.anomavision/
        mvtec/bottle/         <- bottle class images (MVTec layout)
        models/               <- model weights

Usage
-----
    python anomavision_downloader.py

    from anomavision_downloader import ensure_assets, ensure_images, ensure_models
"""

import re
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

# ------------------------------------------------------------------------------
# Terminal colours (work on Windows 10+, Linux, macOS)
# ------------------------------------------------------------------------------


class _C:
    """ANSI colour codes — gracefully disabled if the terminal does not support them."""

    import sys as _sys

    _on = _sys.stdout.isatty()
    RESET = "[0m" if _on else ""
    YELLOW = "[33m" if _on else ""
    RED = "[31m" if _on else ""
    BOLD = "[1m" if _on else ""
    CYAN = "[36m" if _on else ""


def _warn_missing_path(what: str, given: str, using: str, error: bool = False) -> None:
    """
    Print an eye-catching banner when a path is absent or invalid.
    error=True  -> red  (asset missing, user must act)
    error=False -> yellow (auto-download fallback)
    """
    w = _C.RED if error else _C.YELLOW
    r, b, c = _C.RESET, _C.BOLD, _C.CYAN
    icon = (
        "X  AnomaVision - MISSING ASSET" if error else "!  AnomaVision - PATH NOT FOUND"
    )
    if error:
        action_line = f"  Run with : {using}"
    else:
        action_line = "  Action   : downloading / using cached assets"
        action_line += f"\n{w}  Using    : {r}{c}{using}{r}"
    print(
        f"\n{w}{b}"
        "\n  +----------------------------------------------------------+"
        f"\n  |  {icon:<54}|"
        f"\n  +----------------------------------------------------------+{r}"
        f"\n{w}  |  {what} path is not set or does not exist.{r}"
        f"\n{w}  |  Given    : {r}{c}{given or chr(40) + chr(110) + chr(111) + chr(116) + chr(32) + chr(115) + chr(101) + chr(116) + chr(41)}{r}"
        f"\n{w}  |  {action_line}{r}"
        f"\n{w}{b}  +----------------------------------------------------------+{r}\n"
    )


# ------------------------------------------------------------------------------
# CONFIGURATION  <- edit these values
# ------------------------------------------------------------------------------

GITHUB_OWNER: str = "DeepKnowledge1"
GITHUB_REPO: str = "AnomaVision"

# Global cache shared across all installs / virtual environments
CACHE_DIR: Path = Path.home() / ".anomavision"
MVTEC_DIR: Path = CACHE_DIR / "mvtec"  # ~/.anomavision/mvtec/
IMAGES_DIR: Path = MVTEC_DIR / "bottle"  # ~/.anomavision/mvtec/bottle/
MODELS_DIR: Path = CACHE_DIR / "models"  # ~/.anomavision/models/

# GitHub Release tag for models (independent of code releases)
ASSETS_RELEASE_TAG: str = "assets-stable"

# Google Drive file IDs
# How to get: Share file -> "Anyone with the link"
# URL looks like: https://drive.google.com/file/d/<FILE_ID>/view
GDRIVE_IMAGE_FILE_ID: str = "1k5IEqkgBE4i8BoFOK4rh3mmqzq0-VVOg"
GDRIVE_MODEL_FILE_ID: str = (
    "YOUR_MODEL_GDRIVE_FILE_ID_HERE"  # <- upload model_bottle.zip to Drive and paste ID here
)

# Filename patterns for model assets on GitHub Release
# Matches: model_bottle.zip, padim_model.pt, weights_v2.onnx, etc.
MODEL_ASSET_PATTERNS: list[str] = [
    r"model_bottle[.](zip|tar[.]gz|tgz)$",
    r".*model.*[.](pt|pth|onnx|zip|tar[.]gz|tgz)$",
    r".*weights.*[.](pt|pth|onnx|zip|tar[.]gz|tgz)$",
]

# ------------------------------------------------------------------------------

GITHUB_API_ASSETS: str = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
    f"/releases/tags/{ASSETS_RELEASE_TAG}"
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _dir_is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    return not any(path.rglob("*"))


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(re.search(p, name, re.IGNORECASE) for p in patterns)


def _download_file(url: str, dest: Path, params: dict = None) -> Path:
    """Stream-download url to dest with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  -> {dest.name} ... {pct:5.1f}%", end="", flush=True)
    print(f"\r  OK  {dest.name}  ({downloaded / 1_048_576:.1f} MB)")
    return dest


def _extract(archive: Path, target_dir: Path) -> None:
    """Extract a zip or tar.gz into target_dir, then remove the archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    print(f"  -> Extracting {archive.name} ...")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
        archive.unlink()
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target_dir)
        archive.unlink()
    # plain .pt / .onnx — already in place, nothing to do
    print("  OK  Extraction complete.")


# ── Google Drive ───────────────────────────────────────────────────────────────


def _gdrive_download(file_id: str, dest: Path) -> None:
    """
    Download a file from Google Drive.
    Handles Google's large-file virus-scan confirmation (the 'Download anyway' page)
    using the newer /uc?id=...&confirm=t approach that works for large files.
    """
    print("  -> Connecting to Google Drive ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Step 1 — hit the standard export URL
    url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}
    resp = session.get(url, params=params, stream=True, timeout=60)
    resp.raise_for_status()

    # Step 2 — Google returns an HTML confirmation page for large files.
    # Detect this by checking Content-Type and response size.
    content_type = resp.headers.get("Content-Type", "")
    is_html = "text/html" in content_type

    if is_html:
        print("  -> Confirmation page detected — bypassing ...")
        # Read the page to extract the confirm token
        html = resp.content.decode("utf-8", errors="ignore")

        # Try multiple patterns Google has used over the years
        token = None
        for pattern in [
            r"confirm=([0-9A-Za-z_-]+)",
            r"confirm_token=([0-9A-Za-z_-]+)",
            r"confirm=([A-Za-z0-9_-]+)",
        ]:
            m = re.search(pattern, html)
            if m:
                token = m.group(1)
                break

        # Modern Google Drive uses a simpler bypass: confirm=t
        if not token:
            token = "t"

        params["confirm"] = token
        # Also pass the uuid if present (newer Drive requirement)
        uuid_match = re.search(r'"uuid":"([^"]+)"', html)
        if uuid_match:
            params["uuid"] = uuid_match.group(1)

        resp = session.get(url, params=params, stream=True, timeout=300)
        resp.raise_for_status()

        # Verify we now have the actual file, not another HTML page
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            # Last resort: use the direct download URL format
            resp = session.get(
                "https://drive.usercontent.google.com/download",
                params={
                    "id": file_id,
                    "export": "download",
                    "authuser": "0",
                    "confirm": "t",
                },
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

    # Step 3 — stream to disk
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  -> {dest.name} ... {pct:5.1f}%", end="", flush=True)

    print(f"\r  OK  {dest.name}  ({downloaded / 1_048_576:.1f} MB)")

    # Sanity check — make sure we got a real file, not an HTML error page
    if downloaded < 1024:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is too small ({downloaded} bytes) — "
            "likely an error page. Check that the file is shared as "
            "'Anyone with the link' on Google Drive."
        )


# ── GitHub Release ─────────────────────────────────────────────────────────────


def _fetch_assets_release() -> dict:
    """
    Fetch the assets-stable release by scanning all releases and matching
    the tag name exactly. This avoids the GitHub 'Latest' redirect bug where
    releases/tags/<tag> can return a different release if the tag is also
    marked as Latest.
    """
    print(f"  -> Fetching GitHub Release '{ASSETS_RELEASE_TAG}' ...")

    # First try the direct tag endpoint
    resp = requests.get(GITHUB_API_ASSETS, timeout=30)

    if resp.status_code == 200:
        release = resp.json()
        # Verify the tag matches — GitHub can return the wrong release
        if release.get("tag_name") == ASSETS_RELEASE_TAG:
            print(f"  OK  {release['tag_name']} – {release['name']}")
            return release
        else:
            print(
                f"  -> Tag mismatch (got '{release.get('tag_name')}') — scanning all releases ..."
            )

    # Fallback: list all releases and find the correct tag
    list_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
    page = 1
    while True:
        r = requests.get(list_url, params={"per_page": 100, "page": page}, timeout=30)
        r.raise_for_status()
        releases = r.json()
        if not releases:
            break
        for release in releases:
            if release.get("tag_name") == ASSETS_RELEASE_TAG:
                print(
                    f"  OK  {release['tag_name']} – {release['name']} (found via list)"
                )
                return release
        page += 1

    sys.exit(
        f"\n[ERROR] Release tag '{ASSETS_RELEASE_TAG}' not found in "
        f"{GITHUB_OWNER}/{GITHUB_REPO}.\n"
        "        Create it with:\n"
        f"        gh release create {ASSETS_RELEASE_TAG} "
        f"--title 'Stable Assets' --repo {GITHUB_OWNER}/{GITHUB_REPO}"
    )


# Direct download URLs — used as fallback if the GitHub API returns wrong assets
# Format: https://github.com/<owner>/<repo>/releases/download/<tag>/<filename>
_DIRECT_MODEL_ASSETS = [
    "https://github.com/user-attachments/files/26087357/model_bottle.zip",
]
_DIRECT_IMAGE_ASSETS = [
    "https://github.com/user-attachments/files/26087360/sample_bottle_images.zip",
]


def _download_github_models(assets: list[dict], target_dir: Path) -> bool:
    print(f"  -> Release assets found: {[a['name'] for a in assets]}")
    matches = [a for a in assets if _matches_any(a["name"], MODEL_ASSET_PATTERNS)]

    if not matches:
        print(
            "  [WARN] API returned wrong release assets — using direct download URLs ..."
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        for url in _DIRECT_MODEL_ASSETS:
            fname = url.split("/")[-1]
            dest = target_dir / fname
            print(f"  -> Direct: {url}")
            try:
                _download_file(url, dest)
                _extract(dest, target_dir)
            except Exception as e:
                print(f"  [ERROR] Direct download failed: {e}")
                return False
        return True

    for a in matches:
        dest = target_dir / a["name"]
        _download_file(a["browser_download_url"], dest)
        _extract(dest, target_dir)
    return True


def _download_github_images(assets: list[dict], target_dir: Path) -> bool:
    print(f"  -> Release assets found: {[a['name'] for a in assets]}")
    matches = [a for a in assets if _matches_any(a["name"], IMAGE_ASSET_PATTERNS)]

    if not matches:
        print(
            "  [WARN] API returned wrong release assets — using direct download URLs ..."
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        for url in _DIRECT_IMAGE_ASSETS:
            fname = url.split("/")[-1]
            dest = target_dir / fname
            print(f"  -> Direct: {url}")
            try:
                _download_file(url, dest)
                _extract(dest, target_dir)
            except Exception as e:
                print(f"  [ERROR] Direct download failed: {e}")
                return False
        return True

    for a in matches:
        dest = target_dir / a["name"]
        _download_file(a["browser_download_url"], dest)
        _extract(dest, target_dir)
    return True


# ── public API ─────────────────────────────────────────────────────────────────


def ensure_images() -> None:
    """
    Download the full MVTec bottle dataset from Google Drive.
    Used by train.py.
    """
    if not _dir_is_empty(IMAGES_DIR):
        print(f"[AnomaVision] Images already present at {IMAGES_DIR}")
        return

    if (
        not GDRIVE_IMAGE_FILE_ID
        or GDRIVE_IMAGE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
    ):
        sys.exit(
            "\n[ERROR] GDRIVE_IMAGE_FILE_ID is not set in anomavision_downloader.py.\n"
            "        Upload your bottle dataset zip to Google Drive, share it,\n"
            "        and paste the file ID into GDRIVE_IMAGE_FILE_ID."
        )

    print("[AnomaVision] Dataset missing — downloading from Google Drive ...")
    tmp_zip = CACHE_DIR / "bottle_images_tmp.zip"
    try:
        _gdrive_download(GDRIVE_IMAGE_FILE_ID, tmp_zip)
        _extract(tmp_zip, MVTEC_DIR)  # extracts as bottle/ inside mvtec/
        print(f"[AnomaVision] Dataset ready at {IMAGES_DIR}\n")
    except Exception as e:
        if tmp_zip.exists():
            tmp_zip.unlink()
        sys.exit(f"\n[ERROR] Failed to download dataset from Google Drive: {e}")


def ensure_models() -> None:
    """
    Download model_bottle.zip and sample_bottle_images.zip from GitHub.
    Used by detect.py / eval.py / export.py.
    """
    if not _dir_is_empty(MODELS_DIR):
        print(f"[AnomaVision] Models already present at {MODELS_DIR}")
        return

    print("[AnomaVision] Models missing — downloading from GitHub ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for url in _DIRECT_MODEL_ASSETS:
        fname = url.split("/")[-1]
        dest = MODELS_DIR / fname
        print(f"  -> {fname}")
        try:
            _download_file(url, dest)
            _extract(dest, MODELS_DIR)
        except Exception as e:
            sys.exit(f"\n[ERROR] Failed to download {fname}: {e}")

    print(f"[AnomaVision] Models ready at {MODELS_DIR}\n")


def ensure_sample_images() -> None:
    """
    Download sample_bottle_images.zip from GitHub into the MVTec cache.
    Used by detect.py for quick testing without the full dataset.
    """
    sample_dir = MVTEC_DIR / "bottle"
    if not _dir_is_empty(sample_dir):
        print(f"[AnomaVision] Sample images already present at {sample_dir}")
        return

    print("[AnomaVision] Sample images missing — downloading from GitHub ...")
    MVTEC_DIR.mkdir(parents=True, exist_ok=True)

    for url in _DIRECT_IMAGE_ASSETS:
        fname = url.split("/")[-1]
        dest = MVTEC_DIR / fname
        print(f"  -> {fname}")
        try:
            _download_file(url, dest)
            _extract(dest, MVTEC_DIR)  # extracts as bottle/ inside mvtec/
        except Exception as e:
            sys.exit(f"\n[ERROR] Failed to download {fname}: {e}")

    print(f"[AnomaVision] Sample images ready at {sample_dir}\n")


def ensure_assets() -> None:
    """Download both images and models if missing. Call at startup."""
    ensure_images()
    ensure_models()


def find_model_file(model_dir, requested: str):
    """
    Search model_dir for the best available model when the exact
    requested filename does not exist.

    Priority:
      1. Same stem, different extension  (e.g. model.pt -> model.pth)
      2. Any recognised model file in the directory

    Returns the resolved path string, or None if nothing found.
    """
    MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".torchscript"}
    model_dir = Path(model_dir)
    stem = Path(requested).stem

    if not model_dir.is_dir():
        return None

    # Same stem, any model extension
    for ext in MODEL_EXTENSIONS:
        candidate = model_dir / (stem + ext)
        if candidate.exists():
            return str(candidate)

    # Any model file in the directory
    for f in sorted(model_dir.iterdir()):
        if f.suffix.lower() in MODEL_EXTENSIONS:
            return str(f)

    return None


def get_images_dir() -> Path:
    """Returns ~/.anomavision/mvtec/bottle/ — pass as dataset_path to train/eval."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGES_DIR


def get_mvtec_dir() -> Path:
    """Returns ~/.anomavision/mvtec/ — the root containing all MVTec classes."""
    MVTEC_DIR.mkdir(parents=True, exist_ok=True)
    return MVTEC_DIR


def get_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_assets()

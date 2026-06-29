import json
import os
import re
import tempfile
import zipfile
import logging
from datetime import datetime
from pathlib import Path

import requests

from config import cfg

log = logging.getLogger(__name__)
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
METADATA_PATH = MODELS_DIR / "model_sync.json"
PACKAGE_NAME = "models.zip"


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def current_version_name(suffix: str | None = None, now: datetime | None = None) -> str:
    now = now or datetime.utcnow()
    week = now.isocalendar()[1]
    year = now.year % 100
    suffix = suffix or cfg.MODEL_SYNC_VERSION_SUFFIX
    version = f"W{week:02d}Y{year:02d}"
    return f"{version}-{suffix}" if suffix else version


def read_local_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_local_metadata(version: str, files: list[str], source: str = "local", timestamp: str | None = None) -> dict:
    _ensure_models_dir()
    timestamp = timestamp or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    metadata = {
        "version": version,
        "timestamp": timestamp,
        "source": source,
        "files": files,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def package_models(version: str | None = None) -> Path:
    _ensure_models_dir()
    version = version or current_version_name()
    local_files = sorted(p.name for p in MODELS_DIR.glob("*.pkl"))
    if not local_files:
        raise RuntimeError("No model files found in models/ to package.")
    write_local_metadata(version, local_files, source="local")
    zip_path = MODELS_DIR / PACKAGE_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in local_files:
            archive.write(MODELS_DIR / filename, arcname=filename)
        archive.write(METADATA_PATH, arcname=METADATA_PATH.name)
    return zip_path


def _git_repo() -> str | None:
    remote = None
    try:
        import subprocess
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None

    if not remote:
        return None

    m = re.search(r"(?:github\.com[:/])(?P<repo>[^/]+/[^/]+?)(?:\.git)?$", remote)
    return m.group("repo") if m else None


def _github_repo() -> str:
    repo = cfg.GITHUB_REPO or os.getenv("GITHUB_REPO") or _git_repo()
    if not repo:
        raise RuntimeError("GitHub repository not configured. Set GITHUB_REPO in .env or remote origin.")
    return repo


def _github_headers(content_type: str = "application/json") -> dict:
    token = cfg.GITHUB_TOKEN or os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub model sync.")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
    }


def _github_api_url(path: str) -> str:
    return f"https://api.github.com{path}"


def _get_release(tag_name: str) -> dict | None:
    url = _github_api_url(f"/repos/{_github_repo()}/releases/tags/{tag_name}")
    resp = requests.get(url, headers=_github_headers())
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _create_release(tag_name: str, name: str, body: str = "") -> dict:
    url = _github_api_url(f"/repos/{_github_repo()}/releases")
    payload = {
        "tag_name": tag_name,
        "name": name,
        "body": body,
        "draft": False,
        "prerelease": False,
    }
    resp = requests.post(url, json=payload, headers=_github_headers())
    resp.raise_for_status()
    return resp.json()


def _ensure_release(tag_name: str, name: str, body: str = "") -> dict:
    release = _get_release(tag_name)
    if release is not None:
        return release
    return _create_release(tag_name, name, body)


def _delete_asset(asset_id: int) -> None:
    url = _github_api_url(f"/repos/{_github_repo()}/releases/assets/{asset_id}")
    resp = requests.delete(url, headers=_github_headers())
    resp.raise_for_status()


def _upload_asset(release: dict, file_path: Path, asset_name: str) -> dict:
    upload_url = release["upload_url"].split("{")[0]
    existing = [asset for asset in release.get("assets", []) if asset.get("name") == asset_name]
    for asset in existing:
        _delete_asset(asset["id"])
    headers = _github_headers("application/zip")
    headers["Accept"] = "application/vnd.github+json"
    with file_path.open("rb") as file_obj:
        resp = requests.post(
            upload_url,
            params={"name": asset_name},
            headers=headers,
            data=file_obj,
        )
    resp.raise_for_status()
    return resp.json()


def upload_models(version: str | None = None) -> dict:
    if cfg.MODEL_SYNC_METHOD.lower() != "github":
        raise RuntimeError("Only github sync method is implemented currently.")
    version = version or current_version_name()
    zip_path = package_models(version)
    release = _ensure_release(version, version, body=f"Model package {version}")
    asset = _upload_asset(release, zip_path, PACKAGE_NAME)
    log.info("Uploaded model package %s to GitHub release %s", zip_path.name, version)
    return asset


def get_latest_remote_release() -> dict | None:
    url = _github_api_url(f"/repos/{_github_repo()}/releases/latest")
    resp = requests.get(url, headers=_github_headers())
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _download_asset(asset: dict, destination: Path) -> Path:
    url = asset["url"]
    headers = _github_headers()
    headers["Accept"] = "application/octet-stream"
    with requests.get(url, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        with destination.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return destination


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _remote_is_newer(remote_release: dict, local_metadata: dict | None) -> bool:
    if local_metadata is None:
        return True
    remote_ts = _parse_timestamp(remote_release.get("published_at") or remote_release.get("created_at"))
    local_ts = _parse_timestamp(local_metadata["timestamp"])
    return remote_ts > local_ts


def fetch_latest_models() -> bool:
    if cfg.MODEL_SYNC_METHOD.lower() != "github":
        raise RuntimeError("Only github sync method is implemented currently.")
    release = get_latest_remote_release()
    if release is None:
        log.warning("No remote GitHub release found for model sync.")
        return False
    local_metadata = read_local_metadata()
    if not _remote_is_newer(release, local_metadata):
        log.info("Local model version is current; no update needed.")
        return False
    asset = next((a for a in release.get("assets", []) if a.get("name") == PACKAGE_NAME), None)
    if asset is None:
        raise RuntimeError(f"Release {release['tag_name']} has no {PACKAGE_NAME} asset.")
    _ensure_models_dir()
    temp_zip = MODELS_DIR / f"{release['tag_name']}.zip"
    _download_asset(asset, temp_zip)
    with zipfile.ZipFile(temp_zip, "r") as archive:
        archive.extractall(MODELS_DIR)
    temp_zip.unlink(missing_ok=True)
    write_local_metadata(release["tag_name"], [f.name for f in MODELS_DIR.glob("*.pkl")], source="github", timestamp=release.get("published_at"))
    log.info("Fetched new models: %s", release["tag_name"])
    return True


def fetch_latest_models_if_needed() -> bool:
    if not cfg.MODEL_SYNC_AUTO_FETCH:
        return False
    try:
        return fetch_latest_models()
    except Exception as exc:
        log.warning("Model sync fetch failed: %s", exc)
        return False

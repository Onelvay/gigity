"""Resumable, commit-pinned downloads. Raw files retain their source hash."""

import argparse
import gzip
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from touchline.config import ARTIFACTS, COMPETITION, RAW, SEASON, SOURCE_COMMIT, directories


def fetch(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TouchlineLab-research/0.1"})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def download_one(commit, relative, expected_sha=None):
    target = RAW / (relative + ".gz")
    url = f"https://raw.githubusercontent.com/hudl/open-data/{commit}/data/{relative}"
    if target.exists():
        body = gzip.decompress(target.read_bytes())
    else:
        body = fetch(url)
        json.loads(body)  # Never cache an HTML/error response.
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(gzip.compress(body, mtime=0))
        temporary.replace(target)
    digest = hashlib.sha256(body).hexdigest()
    if expected_sha and digest != expected_sha:
        raise ValueError(f"Source checksum mismatch: {relative}. Remove the corrupt cache file and retry.")
    return {"file": relative, "sha256": digest, "bytes": len(body), "url": url}


def read_raw(relative):
    return json.loads(gzip.decompress((RAW / (relative + ".gz")).read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Download a chronological subset for a quick demo")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    directories()
    manifest_path = ARTIFACTS / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        commit = manifest["commit"]
    else:
        commit = SOURCE_COMMIT
        manifest = {"source": "StatsBomb via hudl/open-data", "commit": commit, "files": {}}
    # A cache belongs to a single commit. Do not mix upstream versions.
    for relative in ("competitions.json", f"matches/{COMPETITION}/{SEASON}.json"):
        item = download_one(commit, relative, manifest["files"].get(relative, {}).get("sha256"))
        manifest["files"][relative] = item
    matches = sorted(
        read_raw(f"matches/{COMPETITION}/{SEASON}.json"),
        key=lambda m: (m["match_date"], m["kick_off"], m["match_id"]),
    )
    chosen = matches[: args.limit] if args.limit else matches
    manifest["available_matches"] = len(matches)
    manifest["requested_matches"] = len(chosen)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        jobs = {
            pool.submit(
                download_one,
                commit,
                f"events/{m['match_id']}.json",
                manifest["files"].get(f"events/{m['match_id']}.json", {}).get("sha256"),
            ): m
            for m in chosen
        }
        for count, job in enumerate(as_completed(jobs), 1):
            item = job.result()
            manifest["files"][item["file"]] = item
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if count % 20 == 0 or count == len(chosen):
                print(f"Events cached: {count}/{len(chosen)}", flush=True)
    print(f"Source commit: {commit}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download target-specific model assets from GitHub Releases and verify SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["assets"]:
        target = root / item["path"]
        if target.exists() and hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == item["sha256"]:
            print("Verified " + item["path"])
            continue
        if args.verify_only:
            raise RuntimeError("Missing or invalid asset: " + item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "NemotronNPU/1.0"})
        digest = hashlib.sha256()
        print("Downloading " + item["path"], flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
        if temporary.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise RuntimeError("Download verification failed: " + item["path"])
        temporary.replace(target)
        print("Verified " + item["path"])


if __name__ == "__main__":
    main()

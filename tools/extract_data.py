"""Extract the authorized archive without overwriting existing data."""
import json
import os
import shutil
import stat
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "glaucoma_first.zip"
DEST = ROOT / "data" / "raw"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        for item in entries:
            target = (DEST / item.filename).resolve()
            if DEST.resolve() not in target.parents or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe archive member")
        total = sum(item.file_size for item in entries)
        if shutil.disk_usage(DEST).free < total + 5_000_000_000:
            raise RuntimeError("Insufficient disk space including safety reserve")
        for n, item in enumerate(entries, 1):
            target = DEST / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with archive.open(item) as source, temporary.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            os.replace(temporary, target)
            if n % 1000 == 0:
                print(json.dumps({"extracted": n, "total": len(entries), "seconds": round(time.monotonic()-start)}), flush=True)
    (ROOT / "reports" / "extraction.json").write_text(json.dumps({"entries": len(entries), "uncompressed_bytes": total, "crc_checked_on_read": True, "seconds": time.monotonic()-start}, indent=2))
    print("EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()

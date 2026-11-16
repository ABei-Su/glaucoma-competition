"""Restore byte-identical files listed in the private archive manifest."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f'Path outside extraction directory: {relative}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extraction_directory', type=Path)
    args = parser.parse_args()
    root = args.extraction_directory.resolve(strict=True)
    manifest = Path(__file__).with_name('归档清单.json')
    aliases = json.loads(manifest.read_text(encoding='utf-8'))['duplicate_files']
    verified = set()
    restored = 0
    for relative, entry in aliases.items():
        source = checked_path(root, entry['source'])
        target = checked_path(root, relative)
        if source not in verified:
            if source.stat().st_size != entry['bytes'] or digest(source) != entry['sha256']:
                raise ValueError(f'Archive file failed verification: {source}')
            verified.add(source)
        if target.exists():
            if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
                raise FileExistsError(f'Different file already exists: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            copy = Path(temporary) / target.name
            shutil.copyfile(source, copy)
            if target.exists():
                raise FileExistsError(target)
            copy.rename(target)
        restored += 1
    print(f'Restored {restored} files.')


if __name__ == '__main__':
    main()

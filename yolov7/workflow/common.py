import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def git_commit(path=REPO_ROOT):
    try:
        out = subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '') if out.returncode == 0 else None
    except FileNotFoundError:
        return None


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
        f.write('\n')


def read_json(path):
    with open(path) as f:
        return json.load(f)


def resolve_dataset_yaml(data):
    # Accept a dataset.yaml or an exported dataset version directory containing one
    path = Path(data).expanduser()
    if path.is_dir():
        path = path / 'dataset.yaml'
    if not path.is_file():
        raise SystemExit(f'Dataset not found: {path}')
    return path.absolute()


def print_table(rows, headers):
    rows = [[('' if v is None else str(v)) for v in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*('-' * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def env_path(name, default):
    return Path(os.environ.get(name, default)).expanduser()

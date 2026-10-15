"""yolov7-train: train YOLOv7 on a dataset exported by yolov7-data

Runs yolov7/train.py with the settings used for FlyFinder so far (cfg/training/yolov7.yaml, data/hyp.scratch.p5.yaml,
COCO-pretrained yolov7.pt, batch size 16) and can archive the result in the model registry when it finishes:

    yolov7-train --data /flyhostel_data/fiftyone/FlyFinder/v7 --archive flyfinder

Any argument after `--` is passed to train.py unchanged, e.g. `-- --cache-images --multi-scale`.
"""

import argparse
import datetime
import shlex
import subprocess
import sys
from pathlib import Path

from yolov7.workflow.common import REPO_ROOT, read_json, resolve_dataset_yaml


def default_name(data_yaml):
    manifest = data_yaml.parent / 'manifest.json'
    if manifest.is_file():
        m = read_json(manifest)
        base = f"{m.get('fiftyone_dataset')}_{m.get('version')}"
    else:
        base = data_yaml.parent.name
    return f'{base}_{datetime.date.today().isoformat()}'


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(prog='yolov7-train', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, help='dataset version directory or dataset.yaml')
    parser.add_argument('--name', help='run name (default: <dataset>_<version>_<date>)')
    parser.add_argument('--project', default=str(REPO_ROOT / 'runs' / 'train'), help='directory holding the runs')
    parser.add_argument('--weights', default=str(REPO_ROOT / 'yolov7.pt'),
                        help='initial weights (default: COCO-pretrained yolov7.pt)')
    parser.add_argument('--from-scratch', action='store_true', help='random initial weights instead of --weights')
    parser.add_argument('--cfg', default='cfg/training/yolov7.yaml', help='model definition')
    parser.add_argument('--hyp', default='data/hyp.scratch.p5.yaml', help='hyperparameters')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=36, help='maximum number of dataloader workers')
    parser.add_argument('--archive', metavar='MODEL', help='archive the trained weights as a version of MODEL')
    parser.add_argument('--promote', action='store_true', help='with --archive: make it the production version')
    parser.add_argument('--notes', help='with --archive: notes for the model card')
    parser.add_argument('--registry', help='with --archive: registry directory')
    parser.add_argument('--dry-run', action='store_true', help='print the train.py command without running it')
    args = parser.parse_args(argv)
    args.extra = extra
    return args


def main(argv=None):
    args = parse_args(argv)
    data_yaml = resolve_dataset_yaml(args.data)
    name = args.name or default_name(data_yaml)
    run_dir = Path(args.project) / name
    if run_dir.exists() and '--resume' not in args.extra:
        raise SystemExit(f'{run_dir} already exists: pick another --name')

    if args.from_scratch:
        weights = ''
    else:
        weights = str(Path(args.weights).expanduser().absolute())
        if not Path(weights).is_file():
            raise SystemExit(f'Initial weights not found: {weights} (use --from-scratch to train without them)')
        if not weights.endswith('.pt'):
            raise SystemExit(f'train.py only loads weights ending in .pt: {weights}')

    cmd = [sys.executable, '-m', 'yolov7.train', '--data', str(data_yaml), '--name', name, '--project', args.project,
           '--weights', weights, '--cfg', args.cfg, '--hyp', args.hyp, '--epochs', str(args.epochs),
           '--batch-size', str(args.batch_size), '--img-size', str(args.img_size), '--device', args.device,
           '--workers', str(args.workers), *args.extra]
    print('$ ' + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0

    returncode = subprocess.call(cmd, cwd=REPO_ROOT)
    if returncode != 0:
        print(f'\ntrain.py failed with exit code {returncode}', file=sys.stderr)
        return returncode
    print(f'\nTraining finished: {run_dir}')

    if args.archive:
        from yolov7.workflow import models
        archive_args = ['archive', str(run_dir), '--name', args.archive]
        if args.promote:
            archive_args.append('--promote')
        if args.notes:
            archive_args += ['--notes', args.notes]
        models.main((['--registry', args.registry] if args.registry else []) + archive_args)
    else:
        print(f'Archive it with: yolov7-models archive {run_dir} --name MODEL')
    return 0


if __name__ == '__main__':
    sys.exit(main())

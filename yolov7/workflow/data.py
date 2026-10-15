"""yolov7-data: get training data out of CVAT, through FiftyOne, into versioned YOLO datasets

Annotation tasks are created from FiftyOne datasets (see yolov7tools' cvat-dataset), so FiftyOne knows which CVAT
tasks belong to which samples. The workflow is:

    yolov7-data status --dataset FlyFinder                       # annotation runs, their CVAT tasks, what's pulled
    yolov7-data pull   --dataset FlyFinder --anno-key KEY [KEY ...]
    yolov7-data export --dataset FlyFinder --root /flyhostel_data/fiftyone/FlyFinder

export writes <root>/vN (the next free version) with images/, labels/, dataset.yaml and a manifest.json describing
exactly what went into it. Pass <root>/vN to yolov7-train.
"""

import argparse
import os
import random
import re
import sys
import warnings
from pathlib import Path

import yaml

from yolov7.workflow.common import git_commit, now, print_table, write_json

INFO_KEY = 'yolov7_workflow'  # where pulled annotation runs are recorded in the FiftyOne dataset's info


def load_dataset(name):
    import fiftyone as fo
    if name not in fo.list_datasets():
        raise SystemExit(f"FiftyOne dataset '{name}' not found. Available: {', '.join(fo.list_datasets())}")
    return fo.load_dataset(name)


def pulled_runs(dataset):
    return dict(dataset.info.get(INFO_KEY, {}).get('pulled', {}))


def record_pulled(dataset, anno_key):
    info = dict(dataset.info)
    workflow = dict(info.get(INFO_KEY, {}))
    workflow['pulled'] = {**workflow.get('pulled', {}), anno_key: now()}
    info[INFO_KEY] = workflow
    dataset.info = info
    dataset.save()


class CVATStatus:
    # Status of CVAT tasks (annotation/validation/completed), or 'deleted' if they no longer exist on the server

    def __init__(self):
        import fiftyone as fo
        from cvat_sdk import Client
        cfg = fo.annotation_config.backends.get('cvat', {})
        self.client = Client(cfg.get('url', 'http://localhost:8080'), check_server_version=False)
        self.client.login((cfg.get('username'), cfg.get('password')))

    def __call__(self, task_ids):
        from cvat_sdk.api_client.exceptions import NotFoundException
        status = {}
        for task_id in task_ids:
            try:
                status[task_id] = self.client.tasks.retrieve(task_id).status
            except NotFoundException:
                status[task_id] = 'deleted'
        return status


def cmd_status(args):
    dataset = load_dataset(args.dataset)
    pulled = pulled_runs(dataset)
    keys = dataset.list_annotation_runs()
    keys = sorted(keys, key=lambda k: dataset.get_annotation_info(k).timestamp)[-args.limit:]
    cvat = None
    if not args.no_cvat:
        try:
            cvat = CVATStatus()
        except Exception as e:
            print(f'WARNING: cannot query CVAT ({type(e).__name__}: {e}); task status not shown')
    rows = []
    for key in keys:
        info = dataset.get_annotation_info(key)
        schema = info.config.label_schema or {}
        results = dataset.load_annotation_results(key)
        tasks = ''
        if results is not None and cvat is not None:
            tasks = ', '.join(f'{t}:{s}' for t, s in cvat(results.task_ids).items())
        rows.append([key, info.timestamp.strftime('%Y-%m-%d %H:%M'), len(dataset.load_annotation_view(key)),
                     ','.join(schema), 'yes' if results is not None else 'no (cleaned up)',
                     pulled.get(key, '').replace('T', ' ')[:16], tasks])
    print(f"FiftyOne dataset '{dataset.name}': {len(dataset)} samples, "
          f"{len(dataset.exists(args.label_field))} with '{args.label_field}'\n")
    print_table(rows, ['anno_key', 'created', 'samples', 'fields', 'results stored', 'pulled', 'CVAT tasks'])
    print('\nPull finished runs with: yolov7-data pull --dataset %s --anno-key KEY [KEY ...]' % dataset.name)


def cmd_pull(args):
    dataset = load_dataset(args.dataset)
    pulled = pulled_runs(dataset)
    if args.all_pending:
        keys = [k for k in dataset.list_annotation_runs()
                if k not in pulled and dataset.load_annotation_results(k) is not None]
    else:
        keys = args.anno_key or []
    if not keys:
        raise SystemExit('Nothing to pull: pass --anno-key KEY [KEY ...] or --all-pending (see yolov7-data status)')
    unknown = [k for k in keys if k not in dataset.list_annotation_runs()]
    if unknown:
        raise SystemExit(f'Unknown annotation runs: {unknown}')

    cvat = None if args.mark_only else CVATStatus()
    for key in keys:
        if args.mark_only:
            print(f'{key}: marked as pulled (no annotations downloaded)')
        else:
            # FiftyOne only logs a warning when there is nothing left to download; fail loudly instead
            results = dataset.load_annotation_results(key)
            if results is None:
                raise SystemExit(f'{key}: its results were cleaned up after an earlier download, nothing to pull')
            status = cvat(results.task_ids)
            if all(s == 'deleted' for s in status.values()):
                raise SystemExit(f'{key}: its CVAT tasks {list(status)} no longer exist, nothing to pull '
                                 f'(--mark-only records it as pulled)')
            print(f'{key}: CVAT tasks {status}')
            print(f'{key}: downloading annotations from CVAT ...')
            dataset.load_annotations(key, unexpected=args.unexpected, cleanup=args.cleanup)
            if args.cleanup:
                print(f'{key}: CVAT tasks deleted')
        record_pulled(dataset, key)
    print(f"\n{len(dataset.exists(args.label_field))} samples with '{args.label_field}'. "
          f"Export a new training set with: yolov7-data export --dataset {dataset.name} --root DIR")


def next_version(root):
    versions = [int(m.group(1)) for p in root.glob('v*') if (m := re.fullmatch(r'v(\d+)', p.name)) and p.is_dir()]
    return f'v{max(versions, default=0) + 1}'


def previous_classes(root):
    # Class order must stay the same across versions, otherwise class ids of old and new models disagree
    versions = sorted((int(m.group(1)), p) for p in root.glob('v*') if (m := re.fullmatch(r'v(\d+)', p.name)))
    for _, path in reversed(versions):
        if (path / 'dataset.yaml').is_file():
            with open(path / 'dataset.yaml') as f:
                names = yaml.safe_load(f).get('names')
            return (list(names.values()) if isinstance(names, dict) else list(names)), path
    return None, None


def deduplicate(view, label_field, prefer_tag):
    # Several samples may point to the same image file (e.g. through '/Users/FlySleepLab_Dropbox' and
    # '/Users/FlySleepLab Dropbox'). Export each file once, or it can end up in train and val/test at the same time.
    # Keep the sample with prefer_tag, then one with labels, then the most recently added one.
    groups = {}
    for sample_id, filepath, tags, has_label in zip(view.values('id'), view.values('filepath'), view.values('tags'),
                                                    view.values(label_field)):
        key = (prefer_tag in tags, has_label is not None, sample_id)  # ObjectIds increase with creation time
        groups.setdefault(os.path.realpath(filepath), []).append(key)
    keep = [max(keys)[2] for keys in groups.values()]
    return keep, len(view) - len(keep)


def assign_splits(view, splits, ratios, seed):
    # Keep the split tag a sample already has (so val/test stay fixed across versions), randomly assign the rest
    ids = view.values('id')
    tags = view.values('tags')
    current, untagged = {}, []
    for sample_id, sample_tags in zip(ids, tags):
        found = [s for s in splits if s in sample_tags]
        if len(found) > 1:
            raise SystemExit(f'Sample {sample_id} has more than one split tag: {found}')
        if found:
            current[sample_id] = found[0]
        else:
            untagged.append(sample_id)

    random.Random(seed).shuffle(untagged)
    bounds = [round(sum(ratios[:i + 1]) / sum(ratios) * len(untagged)) for i in range(len(splits))]
    new = {}
    for split, start, stop in zip(splits, [0] + bounds[:-1], bounds):
        new.update({sample_id: split for sample_id in untagged[start:stop]})
    return current, new


def cmd_export(args):
    import fiftyone as fo

    root = Path(args.root).expanduser().absolute()
    version = args.version or next_version(root)
    out = root / version
    if out.exists():
        raise SystemExit(f'{out} already exists; pick another --version')

    classes = args.classes
    if not classes:
        classes, source = previous_classes(root)
        if not classes:
            raise SystemExit(f'No previous version found in {root}: pass --classes in the order you want them indexed')
        print(f'Using classes {classes} from {source}')

    dataset = load_dataset(args.dataset)
    labeled = dataset.exists(args.label_field)
    view = labeled
    if args.background_tag:
        background = dataset.match_tags(args.background_tag).exists(args.label_field, False)
        view = dataset.select(labeled.values('id') + background.values('id'), ordered=False)
    if args.include_tags:
        view = view.match_tags(args.include_tags)
    if args.exclude_tags:
        view = view.match_tags(args.exclude_tags, bool=False)
    if len(view) == 0:
        raise SystemExit('No samples to export')
    keep, n_duplicates = deduplicate(view, args.label_field, args.prefer_tag)
    view = view.select(keep, ordered=False)

    current, new = assign_splits(view, args.splits, args.split_ratios, args.seed)
    assignment = {**current, **new}

    # Summary
    label_counts = view.count_values(f'{args.label_field}.detections.label')
    dropped = {k: v for k, v in label_counts.items() if k not in classes}
    rows = []
    for split in args.splits:
        split_view = view.select([i for i, s in assignment.items() if s == split], ordered=False)
        n_labels = split_view.count_values(f'{args.label_field}.detections.label')
        rows.append([split, len(split_view), sum(1 for i, s in new.items() if s == split),
                     len(split_view.exists(args.label_field, False)), *(n_labels.get(c, 0) for c in classes)])
    print(f"\nExport of FiftyOne dataset '{dataset.name}' ({args.label_field}) -> {out}\n")
    print_table(rows, ['split', 'images', 'newly assigned', 'background', *classes])
    if dropped:
        print(f'\nLabels not in --classes, dropped: {dropped}')
    if n_duplicates:
        print(f'Samples pointing to an image file already exported by another sample, skipped: {n_duplicates}')
    if args.dry_run:
        print('\nDry run: nothing written')
        return

    if new and not args.no_tag:
        for split in args.splits:
            ids = [i for i, s in new.items() if s == split]
            if ids:
                dataset.select(ids).tag_samples(split)
        print(f'\nTagged {len(new)} newly assigned samples with their split, so they stay in it in future versions')

    out.mkdir(parents=True)
    exported = {}
    for split in args.splits:
        ids = [i for i, s in assignment.items() if s == split]
        if not ids:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Ignoring detection with label')
            dataset.select(ids, ordered=False).export(
                export_dir=str(out), dataset_type=fo.types.YOLOv5Dataset, label_field=args.label_field,
                split=split, classes=classes, export_media='symlink' if args.symlink_media else True)
        exported[split] = ids

    # yolov7 resolves train/val paths relative to the cwd, not to 'path': write them as absolute paths
    data = {'path': str(out), **{split: str(out / 'images' / split) for split in exported},
            'nc': len(classes), 'names': list(classes)}
    with open(out / 'dataset.yaml', 'w') as f:
        f.write(f"# Exported by yolov7-data from FiftyOne dataset '{dataset.name}' on {now()}\n")
        yaml.safe_dump(data, f, sort_keys=False)

    write_json(out / 'manifest.json', {
        'version': version,
        'created': now(),
        'fiftyone_dataset': dataset.name,
        'label_field': args.label_field,
        'classes': list(classes),
        'splits': {r[0]: dict(zip(['images', 'newly_assigned', 'background', *classes], r[1:])) for r in rows},
        'dropped_labels': dropped,
        'duplicate_samples_skipped': n_duplicates,
        'include_tags': args.include_tags, 'exclude_tags': args.exclude_tags, 'background_tag': args.background_tag,
        'seed': args.seed,
        'annotation_runs_pulled': pulled_runs(dataset),
        'yolov7_commit': git_commit(),
    })
    write_json(out / 'samples.json', exported)  # FiftyOne sample ids per split
    print(f'\nWrote {out}\nTrain on it with: yolov7-train --data {out}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='yolov7-data', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', required=True, help='FiftyOne dataset name, e.g. FlyFinder')
    common.add_argument('--label-field', default='ground_truth', help='FiftyOne field holding the annotations')

    p = sub.add_parser('status', parents=[common], help='list annotation runs and the state of their CVAT tasks')
    p.add_argument('--limit', type=int, default=20, help='show only the most recent runs')
    p.add_argument('--no-cvat', action='store_true', help='do not query the CVAT server for task status')
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('pull', parents=[common], help='download annotations of CVAT tasks into the FiftyOne dataset')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--anno-key', nargs='+', help='annotation run(s) to pull, see yolov7-data status')
    g.add_argument('--all-pending', action='store_true', help='pull every run with stored results not yet pulled')
    p.add_argument('--unexpected', default='prompt', choices=['prompt', 'ignore', 'keep', 'return'],
                   help='what to do with labels CVAT returns that are not in the run schema (fiftyone semantics)')
    p.add_argument('--cleanup', action='store_true', help='delete the CVAT tasks after downloading (irreversible)')
    p.add_argument('--mark-only', action='store_true',
                   help='only record the run(s) as pulled, e.g. for runs pulled before this tool existed')
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser('export', parents=[common], help='export the annotations as a new versioned YOLO dataset')
    p.add_argument('--root', required=True, help='directory holding the dataset versions (v1, v2, ...)')
    p.add_argument('--version', help='version name (default: next vN under --root)')
    p.add_argument('--classes', nargs='+', help='class names in index order (default: same as the latest version)')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='split tags')
    p.add_argument('--split-ratios', nargs='+', type=float, default=[0.7, 0.15, 0.15],
                   help='ratios used to assign samples that have no split tag yet')
    p.add_argument('--seed', type=int, default=0, help='random seed of the split assignment')
    p.add_argument('--no-tag', action='store_true', help='do not save the split assignment as tags in FiftyOne')
    p.add_argument('--background-tag', default='background',
                   help="also export samples with this tag and no labels, as background images ('' to disable)")
    p.add_argument('--include-tags', nargs='+', help='only export samples with any of these tags')
    p.add_argument('--exclude-tags', nargs='+', help='do not export samples with any of these tags')
    p.add_argument('--prefer-tag', default='validated',
                   help='when several samples point to the same image file, export the one with this tag')
    p.add_argument('--symlink-media', action='store_true', help='symlink images instead of copying them')
    p.add_argument('--dry-run', action='store_true', help='show what would be exported without writing anything')
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    if getattr(args, 'split_ratios', None) and len(args.split_ratios) != len(args.splits):
        parser.error('--split-ratios needs one value per split')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())

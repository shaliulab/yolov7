# Training workflow: CVAT → dataset → model → registry → CVAT

Three commands, installed with the package (`pip install -e .` in the `yolov7` conda env). Each has `--help`.

## 1. Get new annotations out of CVAT

Annotation tasks are created from FiftyOne datasets (`cvat-dataset` in yolov7tools), so FiftyOne knows which CVAT
task belongs to which sample.

```bash
yolov7-data status --dataset FlyFinder                          # annotation runs, CVAT task status, what's pulled
yolov7-data pull   --dataset FlyFinder --anno-key anno_20261015_101500
yolov7-data export --dataset FlyFinder --root /flyhostel_data/fiftyone/FlyFinder   # writes the next vN
```

`export` writes `<root>/vN/` with `images/`, `labels/`, `dataset.yaml` (absolute paths, as yolov7 needs) and a
`manifest.json` recording the classes, the counts per split and which annotation runs were included.

- Class order is copied from the previous version unless `--classes` is given, so class ids stay compatible.
- Samples keep the `train`/`val`/`test` tag they have. Samples without one are assigned randomly (`--split-ratios`,
  `--seed`) and tagged in FiftyOne, so val/test stay fixed in later versions (`--no-tag` to opt out).
- Samples pointing to the same image file are exported once, so an image can't be in both train and val/test.
- Samples tagged `background` without labels are exported as background images.
- `--dry-run` shows the counts without writing anything.

## 2. Train

```bash
yolov7-train --data /flyhostel_data/fiftyone/FlyFinder/v7 --archive flyfinder
```

Same settings as `train.sh` (cfg/training/yolov7.yaml, data/hyp.scratch.p5.yaml, batch 16, COCO-pretrained
`yolov7.pt`), 400 epochs. The run goes to `runs/train/<dataset>_<version>_<date>`. Arguments after `--` go straight to
`train.py`. `--archive` runs step 3 when training succeeds. `train.py` itself is still called exactly as before.

## 3. Archive and manage weights

```bash
yolov7-models archive runs/train/FlyFinder_v7_2026-10-15 --name flyfinder --notes "adds Oct 2026 annotations"
yolov7-models list
yolov7-models show flyfinder                         # model card of the production version
yolov7-models promote flyfinder FlyFinder_v7_2026-10-15
python yolov7/detect.py --weights $(yolov7-models path flyfinder) ...
```

The registry is `$YOLOV7_REGISTRY`, or `registry/` in this repository:

```
registry/
  README.md                         index of all models and versions (regenerated, do not edit)
  flyfinder/
    production -> FlyFinder_v7_2026-10-15
    HISTORY.log                     archive / promote events
    FlyFinder_v7_2026-10-15/
      weights.pt                    stripped FP16 checkpoint (~75MB), what detect.py loads
      model.yaml                    classes, metrics at the best epoch, dataset, training settings, sha256, source run
      train/                        opt.yaml, hyp.yaml, results.txt and plots
      dataset/                      dataset.yaml and manifest.json the model was trained on
```

A version is self-contained, so once a run is archived its `weights/` directory (often >10GB of intermediate
checkpoints) is no longer needed by anything.

## 4. Pre-populate CVAT identity tasks with detections

```bash
yolov7-cvat predict 2366 2367 --dry-run       # detect and link, write logs/cvat/task_<id>_dry-run.json, upload nothing
yolov7-cvat predict 2366 2367                 # flyfinder3.0 (pinned, not production), conf 0.5; --model to change
yolov7-cvat predict --project FlyHostel3_2X_2026-10-06_13-00-00
```

Every detected fly becomes a CVAT track labelled `UNASSIGNED` (added to the project if missing), with a box on every frame.
Detections that clearly belong to the same fly in consecutive frames of a clip share one track, so in CVAT:

- give one box of a track an identity (1, 2, ...) and the whole track gets it;
- a track ends where linking is ambiguous (flies touching, a fly jumping, appearing or disappearing): give the next
  track the same identity; if a track ever follows the wrong fly, Split it (Alt+M) at the swap;
- leave anything you do not want as `UNASSIGNED`: flyhostel ignores that label when it reads the annotations
  (`UNASSIGNED_LABEL` in `flyhostel/data/human_validation/cvat/cvat_integration.py`).

Detection runs on the raw video frames (CVAT frames are 1000x1000 copies with overlays) when they are on disk and
show the flies where the CVAT frame does, otherwise on the CVAT frame; blank separator frames are skipped. Tasks
that already have annotations are skipped unless `--append`. Needs `$CVAT_USERNAME` / `$CVAT_PASSWORD`.

### Give the tracks their identities

```bash
yolov7-cvat predict --assign --experiment FlyHostel1_3X_2026-09-22_14-00-00   # predict, then assign
yolov7-cvat assign 2370 --dry-run          # on a task already pre-populated: report only (logs/cvat/task_<id>_assign_dry-run.csv)
yolov7-cvat assign 2370
```

`--experiment` takes the tasks named after the experiment (as `upload_to_cvat.sh` names them) and those in a project
named after it. `assign` does what a validator does, clip by clip and chunk by chunk, from the experiment's `.db`
(fragments and local identities of idtrackerai): a track gets an identity (source `semi-auto`) where idtrackerai has
the fly missing or wrong and the fragment was not already labelled earlier in the chunk; a fly the predictions missed
gets a box when idtrackerai or a lower-confidence detection has it. Everything a person still has to do is a CVAT
issue on its frame (Issues in the job's side panel), and a row of `logs/cvat/task_<id>_assign.csv`:

- *check the label / the box added*: medium-confidence changes;
- *label frames ... (best guess N)*: unclear cases, left `UNASSIGNED`;
- *add box for fly N*: nothing found for a fly, usually a take-off or jump where the fly is a motion-blur streak.

It only runs on untouched pre-populated tasks (every track `UNASSIGNED` and `auto`), and saves the annotations to
`logs/cvat/task_<id>_assign_<time>_backup.json` first. Single-fly experiments are skipped. In the validation pipeline,
`vsc-scripts/manual/validation/prepopulate_cvat.sh <experiment>` runs it after `upload_to_cvat.sh`.

## Checkpoint compatibility

Checkpoints trained before this code became the `yolov7` package (including `yolov7.pt` and
`runs/train/flyfinder2.0`) pickle their classes as `models.yolo.Model`. They now load without the
`ln -fs .../yolov7/models .` symlink (`yolov7/utils/checkpoint.py`). Archived and newly trained weights reference
`yolov7.models.*` and load with any checkout of this package, old or new.

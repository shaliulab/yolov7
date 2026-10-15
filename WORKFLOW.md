# Training workflow: CVAT → dataset → model → registry

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

## Checkpoint compatibility

Checkpoints trained before this code became the `yolov7` package (including `yolov7.pt` and
`runs/train/flyfinder2.0`) pickle their classes as `models.yolo.Model`. They now load without the
`ln -fs .../yolov7/models .` symlink (`yolov7/utils/checkpoint.py`). Archived and newly trained weights reference
`yolov7.models.*` and load with any checkout of this package, old or new.

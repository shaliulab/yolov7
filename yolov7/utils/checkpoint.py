# Checkpoint loading that works regardless of the module path classes were pickled under

import contextlib
import importlib
import sys

import torch

# Checkpoints saved before this code became the `yolov7` package (including the upstream yolov7.pt and
# runs/train/flyfinder2.0) pickle their classes as models.yolo.Model, models.common.Conv, ...
# Unpickling them used to require a `models` symlink in the working directory; instead, alias those
# names to the package modules while loading.
_LEGACY_MODULES = ('models', 'models.common', 'models.experimental', 'models.yolo')


@contextlib.contextmanager
def legacy_module_aliases():
    added = []
    for name in _LEGACY_MODULES:
        if name not in sys.modules:
            sys.modules[name] = importlib.import_module(f'yolov7.{name}')
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


def load_checkpoint(path, map_location=None):
    with legacy_module_aliases():
        return torch.load(path, map_location=map_location)

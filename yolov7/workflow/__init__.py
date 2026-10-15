# Command line workflow around YOLOv7: CVAT/FiftyOne data -> training -> archived models
#
#   yolov7-data    pull annotations from CVAT into FiftyOne and export versioned YOLO datasets
#   yolov7-train   train a model on an exported dataset
#   yolov7-models  archive trained weights in a model registry and manage which version is in production

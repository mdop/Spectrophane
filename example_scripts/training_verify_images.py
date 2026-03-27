import json

from spectrophane.pipeline.training_pipeline import resolve_training_paths
from spectrophane.training.ingest_images import calibration_images_with_rois


calibration_filename = "training_data/default.json"
calibration_file, _ = resolve_training_paths(calibration_filename, None, True, True)
with calibration_file.open("r") as f:
    calibration_data = json.load(f)
for img in calibration_images_with_rois(calibration_data):
    img.show()
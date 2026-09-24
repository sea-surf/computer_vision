from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).parent.parent

model_path = ROOT / "models" / "fire_yolo26n.engine"
predict_dir = ROOT / "predict" / "fire_yolo26n"
dataset_path = ROOT / "dataset" / "urban_fire"

model = YOLO(str(model_path))

print(f"Start inference: {video_path}")
results = model.predict(
    source=str(video_path),
    save=True,
    stream=True,
    project=str(predict_dir),
    name="urban_fire",
    conf=0.5
)

for r in results:
    pass

print(f"Inference saved: {predict_dir / 'fire_yolo26n'}")

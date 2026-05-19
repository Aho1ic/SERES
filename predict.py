from ultralytics import YOLO
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SOLARPANEL_MODEL_PATH, PROJECT_ROOT

model = YOLO(SOLARPANEL_MODEL_PATH)
results = model.predict(
    source=str(PROJECT_ROOT / 'data' / '2号门停车场'),
    project=str(PROJECT_ROOT / 'result' / '2号门停车场'),
    name='photography',
    save=True,
    save_txt=False,
    save_conf=False,
    exist_ok=True,
    conf=0.5,
    imgsz=1280
)

print(f"结果已保存至: ./result/photography")
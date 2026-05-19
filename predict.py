from ultralytics import YOLO

model = YOLO('/home/algorithm/chongqing/weights/solarpanel.pt')
results = model.predict(
    source='/home/algorithm/chongqing/data/2号门停车场',
    project='/home/algorithm/chongqing/result/2号门停车场',
    name='photography',
    save=True,
    save_txt=False,
    save_conf=False,
    exist_ok=True,
    conf=0.5,
    imgsz=1280
)

print(f"结果已保存至: ./result/photography")
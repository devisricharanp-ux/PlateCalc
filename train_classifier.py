from ultralytics import YOLO

if __name__ == '__main__':
    for macro_id in range(12):
        model = YOLO('yolov8n-cls.pt')
        model.train(
            data=rf"pathtocategory_{macro_id}",
            epochs=30,
            imgsz=224,
            batch=32,
            device=0,
            amp=True,
            workers=4,
            project="cls_models",
            name=f"category_{macro_id}",
        )

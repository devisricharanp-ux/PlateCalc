if __name__ == '__main__':
    from ultralytics import YOLO
    import torch
    import gc

    # Force a clear of any cached garbage in Python/CUDA before starting
    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO("yolov8s-seg.pt")     #Can also use nano model since there are just 12 categories to remember
    
    results = model.train(
        data="config_macro.yaml", 
        epochs=50, 
        imgsz=640,
        batch=16,          # try 16 first, drop to 8 if OOM
        device=0,
        workers=4,
        amp=True,          # mixed precision - huge VRAM saver, minimal accuracy loss
        cache=True,        # cache images in RAM if dataset fits (speeds up a lot)
        patience=15 
                    )

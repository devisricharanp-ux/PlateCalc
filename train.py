if __name__ == '__main__':
    from ultralytics import YOLO
    import torch
    import gc

    # Force a clear of any cached garbage in Python/CUDA before starting
    gc.collect()
    torch.cuda.empty_cache()

    model = YOLO("yolov8s-seg.pt") 
    
    results = model.train(
        data="config.yaml", 
        epochs=100, 
        batch=16,          # Lowering to 16 completely stabilizes VRAM on an 8GB card
        imgsz=640,         
        workers=2,         # Keeps CPU dataloader system RAM light
        cache=False,       
        overlap_mask=True, # Forces overlapping masks onto one canvas to save memory
        max_det=50,        # Reduces max evaluation objects per image from 300 to 50
        val=True,          # Keep validation enabled
        plots=True         # Disabling validation plots stops massive tensor accumulation in VRAM
    )

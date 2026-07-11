from ultralytics import YOLO
import cv2

flat_model_path = r"C:\Users\devis\OneDrive\Desktop\PlateCalc\runs\segment\train_small\weights\best.pt"  # adjust to your actual path
flat_model = YOLO(flat_model_path)
image_path = r"C:\Users\devis\OneDrive\Pictures\Screenshots\test1.png"  # adjust to your actual image path

image = cv2.imread(image_path) 

flat_results = flat_model(image)[0]

print("\n--- Flat 103-class model results ---")
for i, box in enumerate(flat_results.boxes):
    cls_id = int(box.cls.item())
    cls_name = flat_results.names[cls_id]
    conf = float(box.conf.item())
    print(f"  {cls_name}: {conf:.2f}")
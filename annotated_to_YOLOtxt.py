import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_foodseg_to_yolo(ann_dir, output_txt_dir):
    """
    Converts FoodSeg103 png annotation masks into YOLOv8 segmentation txt files.
    - Drops background (0)
    - Shifts food categories down by 1 (1-103 becomes 0-102)
    - Normalizes polygon coordinates
    """
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    # Get all annotation png files
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.png')]
    
    print(f"Found {len(ann_files)} mask files to convert in {ann_dir}...")

    for file_name in tqdm(ann_files):
        mask_path = os.path.join(ann_dir, file_name)
        
        # Load the mask as grayscale/index array
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        height, width = mask.shape
        unique_classes = np.unique(mask)
        
        yolo_lines = []
        
        for current_class in unique_classes:
            # Skip class 0 (background) completely
            if current_class == 0:
                continue
                
            # YOLO class ID must start at 0 (shift down by 1)
            yolo_class_id = int(current_class - 1)
            
            # Create a isolated binary mask for just this specific food class
            class_mask = np.where(mask == current_class, 255, 0).astype(np.uint8)
            
            # Find the polygon contours of the food shape
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter out tiny accidental artifact fragments (less than 5 pixels)
                if len(contour) < 5:
                    continue
                    
                # Flatten the contour array and normalize coordinates between 0 and 1
                normalized_points = []
                for point in contour:
                    x = point[0][0] / width
                    y = point[0][1] / height
                    normalized_points.append(f"{x:.6f} {y:.6f}")
                
                # Format: <class_id> <x1> <y1> <x2> <y2> ...
                line_str = f"{yolo_class_id} " + " ".join(normalized_points)
                yolo_lines.append(line_str)
                
        # Define output txt file path (same filename but changing extension to .txt)
        txt_filename = os.path.splitext(file_name)[0] + ".txt"
        output_path = os.path.join(output_txt_dir, txt_filename)
        
        # Write the text lines to the file
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines) + "\n")

if __name__ == "__main__":
    # ------------------ CONFIGURATION ------------------
    # Update these paths to match your local dataset folders
    TRAIN_ANN_DIR = 
    VAL_ANN_DIR   = 
    
    TRAIN_OUT_DIR = 
    VAL_OUT_DIR   = 
    # ---------------------------------------------------

    print("--- Converting Training Dataset Masks ---")
    convert_foodseg_to_yolo(TRAIN_ANN_DIR, TRAIN_OUT_DIR)
    
    print("\n--- Converting Validation Dataset Masks ---")
    convert_foodseg_to_yolo(VAL_ANN_DIR, VAL_OUT_DIR)
    
    print("\nAll conversions finished successfully! Labels match the 103-category setup.")

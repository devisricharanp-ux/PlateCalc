from ultralytics import YOLO
import cv2
import numpy as np
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv(#add path to .env file here)
API_KEY = os.getenv("USDA_API_KEY")

COUNTABLE = [
    "egg tart", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry",
    "cherry", "raspberry", "mango", "peach", "lemon", "pear", "fig", "pineapple", "kiwi",
    "orange", "crab", "shrimp", "corn", "hanamaki baozi", "wonton dumplings",
]

SEARCH_OVERRIDE = {
    "chicken duck":     "chicken meat cooked",
    "fried meat":       "fried beef",
    "hanamaki baozi":   "steamed bun",
    "wonton dumplings": "wonton",
    "french fries":     "french fried potato",
    "cheese butter":    "butter",
    "spring onion":     "scallions",
    "rape":             "rapeseed",
    "French beans":     "green snap beans",
    "cilantro mint":    "cilantro",
    "other ingredients": None,
    "sauce":            None,
}

FIXED_WEIGHTS = {
    "egg tart":          80,
    "egg":               50,
    "apple":            182,
    "date":               8,
    "apricot":           35,
    "avocado":          200,
    "banana":           118,
    "strawberry":        12,
    "cherry":             8,
    "raspberry":          5,
    "mango":            200,
    "peach":            150,
    "lemon":            100,
    "pear":             178,
    "fig":               50,
    "pineapple":        905,
    "kiwi":              75,
    "orange":           130,
    "crab":             100,
    "shrimp":            10,
    "corn":              90,
    "hanamaki baozi":    50,
    "wonton dumplings":  20,
}

# --- USDA lookup, fetched lazily and cached per-class ---
CALORIE_CACHE = {}

def get_calories_usda(food_name):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_name,
        "dataType": ["SR Legacy"],
        "pageSize": 1,
        "api_key": API_KEY,
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if not data.get("foods"):
            return None
        food = data["foods"][0]
        for nutrient in food["foodNutrients"]:
            if nutrient["nutrientName"] == "Energy" and nutrient["unitName"] == "KCAL":
                return nutrient["value"]
    except Exception as e:
        print(f"API error for {food_name}: {e}")
    return None

def get_calories_cached(cls_name):
    """Look up kcal/100g for a detected class, hitting the USDA API only
    once per class and reusing the result for repeat detections."""
    if cls_name in CALORIE_CACHE:
        return CALORIE_CACHE[cls_name]

    search_term = SEARCH_OVERRIDE.get(cls_name, cls_name)
    if search_term is None:
        CALORIE_CACHE[cls_name] = 100  # default fallback, no API call needed
        return CALORIE_CACHE[cls_name]

    kcal = get_calories_usda(search_term)
    CALORIE_CACHE[cls_name] = kcal if kcal else 100
    print(f"  {cls_name:30s} -> {CALORIE_CACHE[cls_name]} kcal/100g")
    time.sleep(0.3)
    return CALORIE_CACHE[cls_name]

# --- Weight & Calorie estimation ---
TOTAL_IMAGE_PIXELS = 512 * 403
PLATE_COVERAGE     = 0.70
PLATE_PIXELS       = TOTAL_IMAGE_PIXELS * PLATE_COVERAGE
TYPICAL_MEAL_WEIGHT_G = 400
WEIGHT_PER_PIXEL   = TYPICAL_MEAL_WEIGHT_G / PLATE_PIXELS

print(f"WEIGHT_PER_PIXEL = {WEIGHT_PER_PIXEL:.5f} g/pixel\n")

def estimate_weight(cls_name, mask_pixel_count, instance_count=1):
    if cls_name in COUNTABLE:
        return FIXED_WEIGHTS.get(cls_name, 50) * instance_count
    else:
        return round(mask_pixel_count * WEIGHT_PER_PIXEL, 1)

def estimate_calories(cls_name, weight_g):
    kcal_per_100g = get_calories_cached(cls_name)
    return round((weight_g / 100) * kcal_per_100g, 1)

# --- Load model and image ---
model_path = os.getenv("MODEL_DIR")
image_path = os.getenv("IMG_DIR")

model = YOLO(model_path)
image = cv2.imread(image_path)
H, W, _ = image.shape

results = model(image)

for result in results:
    masks   = result.masks.data
    classes = result.boxes.cls
    confs   = result.boxes.conf

    # Resize all masks
    masks_resized = []
    for mask in masks:
        mask_np      = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (W, H))
        masks_resized.append((mask_resized > 0.5).astype(np.uint8))

    # Remove duplicate masks using IoU
    suppressed = set()
    for i in range(len(masks_resized)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(masks_resized)):
            if j in suppressed:
                continue
            if classes[i] != classes[j]:
                continue
            intersection = np.logical_and(masks_resized[i], masks_resized[j]).sum()
            union        = np.logical_or(masks_resized[i],  masks_resized[j]).sum()
            iou          = intersection / union if union > 0 else 0
            if iou > 0.5:
                if confs[i] >= confs[j]:
                    suppressed.add(j)
                else:
                    suppressed.add(i)

    # Process non-suppressed masks
    total_calories = 0
    print("\n====== Detection Results ======")

    for i, mask_binary in enumerate(masks_resized):
        if i in suppressed:
            continue

        cls_id   = int(classes[i].item())
        cls_name = result.names[cls_id]
        conf     = float(confs[i].item())

        # CCA for instance count
        num_components, _, stats, _ = cv2.connectedComponentsWithStats(mask_binary)
        instance_count = sum(
            1 for c in range(1, num_components)
            if stats[c, cv2.CC_STAT_AREA] > 100
        )

        mask_pixel_count = np.sum(mask_binary)
        weight_g         = estimate_weight(cls_name, mask_pixel_count, instance_count)
        calories         = estimate_calories(cls_name, weight_g)
        total_calories  += calories

        cv2.imwrite(rf"mask_{i}.png", mask_binary * 255)

        print(f"  {cls_name:25s} | conf: {conf:.2f} | "
              f"instances: {instance_count} | "
              f"weight: {weight_g:.1f}g | "
              f"calories: {calories:.1f} kcal")

    print(f"\n  TOTAL CALORIES: {total_calories:.1f} kcal")
    print("================================\n")

import os
from PIL import Image

# ====== CONFIGURATION ======
IMAGES_DIR =   # <-- set your images directory here
LABELS_DIR =   # <-- set your labels directory here
OUTPUT_DIR =  # <-- set your output directory here

TARGET_CATEGORIES = [51, 78, 88, 102]  # <-- set your category IDs here

#example -->
CATEGORY_NAMES = {
    51: "sauce",
    78: "okra",
    88: "cilantro mint",
    102: "other ingredients"
}

# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue

    base_name = os.path.splitext(label_file)[0]

    image_path = None
    for ext in image_extensions:
        candidate = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"Image not found for label: {label_file}")
        continue

    label_path = os.path.join(LABELS_DIR, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        continue

    img = Image.open(image_path)
    img_w, img_h = img.size

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])

        if cls_id not in TARGET_CATEGORIES:
            continue

        coords = list(map(float, parts[1:]))

        if len(coords) == 4:
            # bounding box format: x_center, y_center, w, h
            x_center, y_center, w, h = coords
            x_center *= img_w
            y_center *= img_h
            w *= img_w
            h *= img_h
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
        else:
            # polygon/segmentation format: x1,y1,x2,y2,...
            xs = coords[0::2]
            ys = coords[1::2]
            x1 = min(xs) * img_w
            x2 = max(xs) * img_w
            y1 = min(ys) * img_h
            y2 = max(ys) * img_h

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        cropped = img.crop((x1, y1, x2, y2))

        category_name = CATEGORY_NAMES.get(cls_id, f"class_{cls_id}")
        category_folder = os.path.join(OUTPUT_DIR, category_name)
        os.makedirs(category_folder, exist_ok=True)

        save_name = f"{base_name}_{idx}.jpg"
        save_path = os.path.join(category_folder, save_name)

        cropped.save(save_path)
        print(f"Saved: {save_path}")

print("Done! Crops saved in:", OUTPUT_DIR)

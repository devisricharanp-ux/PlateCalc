# Create an indexed list matching your FoodSeg103 labels to get the original ID indices
original_classes = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter",
    "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew",
    "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana",
    "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
    "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage",
    "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn",
    "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant",
    "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra",
    "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint",
    "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom",
    "oyster mushroom", "white button mushroom", "salad", "other ingredients"
]

import os
import shutil

# --- Step 1: Define your macro category map (0-11), background excluded ---
MACRO_MAP = {
    # Category 0: Grains & Starchy Mains
    "bread": 0, "hamburg": 0, "pizza": 0, "hanamaki baozi": 0, " hanamaki baozi": 0,
    "wonton dumplings": 0, "pasta": 0, "noodles": 0, "rice": 0, "pie": 0, "corn": 0,

    # Category 1: Lean Proteins & Seafood
    "egg": 1, "chicken duck": 1, "fish": 1, "tofu": 1,
    "crab": 1, "shellfish": 1, "shrimp": 1,

    # Category 2: Fatty Meats
    "steak": 2, "pork": 2, "sausage": 2, "fried meat": 2, "lamb": 2,

    # Category 3: Fried & Crispy Snacks
    "french fries": 3, "popcorn": 3, "egg tart": 3,

    # Category 4: Leafy & Low-Calorie Veg
    "eggplant": 4, "cauliflower": 4, "tomato": 4, "kelp": 4, "seaweed": 4,
    "rape": 4, "lettuce": 4, "cucumber": 4, "broccoli": 4, "celery stick": 4,
    "snow peas": 4, "cabbage": 4, " cabbage": 4, "bean sprouts": 4,
    "green beans": 4, "French beans": 4, "salad": 4, "asparagus": 4, "bamboo shoots": 4,

    # Category 5: Root & Allium Veg
    "potato": 5, "pumpkin": 5, "white radish": 5, "carrot": 5,
    "onion": 5, "garlic": 5, "ginger": 5, "spring onion": 5, "pepper": 5,

    # Category 6: Fungi
    "king oyster mushroom": 6, "shiitake": 6, "enoki mushroom": 6,
    "oyster mushroom": 6, "white button mushroom": 6,

    # Category 7: Fruits
    "apple": 7, "apricot": 7, "banana": 7, "strawberry": 7, "cherry": 7,
    "blueberry": 7, "raspberry": 7, "mango": 7, "peach": 7, "lemon": 7,
    "pear": 7, "fig": 7, "pineapple": 7, "grape": 7, "kiwi": 7, "melon": 7,
    "orange": 7, "watermelon": 7, "avocado": 7, "olives": 7,

    # Category 8: Nuts, Seeds & Legumes
    "almond": 8, "red beans": 8, "cashew": 8, "dried cranberries": 8,
    "soy": 8, "walnut": 8, "peanut": 8, "date": 8,

    # Category 9: Sweets & Desserts
    "candy": 9, "chocolate": 9, "biscuit": 9, "pudding": 9, "ice cream": 9, "cake": 9,

    # Category 10: Beverages & Dairy
    "cheese butter": 10, "wine": 10, "milkshake": 10, "coffee": 10,
    "juice": 10, "milk": 10, "tea": 10, "soup": 10,

    # Category 11: Condiments & Mixed/Aromatics
    "sauce": 11, "okra": 11, "cilantro mint": 11, "other ingredients": 11,

    # Background -> sentinel, will be filtered out
    "background": -1,
}

# --- Step 2: original_classes must match your dataset's data.yaml order exactly ---
# Example (REPLACE with your actual list loaded from data.yaml):
# original_classes = ["background", "candy", "egg tart", ..., "other ingredients"]

# Build id -> macro_id map, with safe fallback for anything not in MACRO_MAP
id_conversion = {}
for i, name in enumerate(original_classes):
    if name not in MACRO_MAP:
        raise KeyError(f"Class id {i} ('{name}') not found in MACRO_MAP. "
                        f"Check spelling/whitespace against original_classes.")
    id_conversion[i] = MACRO_MAP[name]

print("ID conversion map:", id_conversion)


def remap_yolo_labels(src_folder, dst_folder, id_conversion):
    """
    Reads YOLO label .txt files from src_folder, remaps class ids using
    id_conversion, drops any boxes mapped to -1 (e.g. background), and
    writes the result to dst_folder. Does NOT modify src_folder.
    """
    os.makedirs(dst_folder, exist_ok=True)
    dropped_files_emptied = 0
    total_boxes_dropped = 0

    for filename in os.listdir(src_folder):
        if not filename.endswith('.txt'):
            continue

        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        with open(src_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            old_id = int(parts[0])
            new_id = id_conversion.get(old_id)

            if new_id is None:
                raise KeyError(f"Class id {old_id} in {filename} not found in id_conversion. "
                                f"Your original_classes list may not match the dataset.")

            if new_id == -1:
                # Skip background/unmapped boxes
                total_boxes_dropped += 1
                continue

            parts[0] = str(new_id)
            new_lines.append(" ".join(parts) + "\n")

        if not new_lines:
            dropped_files_emptied += 1

        with open(dst_path, 'w') as f:
            f.writelines(new_lines)

    print(f"Done. Wrote remapped labels to: {dst_folder}")
    print(f"Boxes dropped (background/unmapped): {total_boxes_dropped}")
    print(f"Files now empty after remap: {dropped_files_emptied}")


# --- Step 3: Run on a COPY of the labels, never the original ---
ORIGINAL_LABELS = r"C:\Users\devis\OneDrive\Desktop\FoodCal\FoodSeg103\ann_dir\labels\val"  # <-- set your original labels directory here
REMAPPED_LABELS = r"C:\Users\devis\OneDrive\Desktop\FoodCal\FoodSeg103\ann_dir\labels\val_remapped"  # <-- set your remapped labels directory here

# Optional: keep an untouched backup of the originals too
# shutil.copytree(ORIGINAL_LABELS, "path/to/backup/labels_original", dirs_exist_ok=True)

remap_yolo_labels(ORIGINAL_LABELS, REMAPPED_LABELS, id_conversion)

# Execute the transformation


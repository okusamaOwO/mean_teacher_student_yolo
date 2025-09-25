import os

images_dir = "mydataset/foggy_zurich_direct/train/images"
labels_dir = "mydataset/foggy_zurich_direct/train/labels"

# Create labels folder if not exists
os.makedirs(labels_dir, exist_ok=True)

# Loop through images
for filename in os.listdir(images_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() in [".jpg", ".jpeg", ".png"]:  # only image files
        label_path = os.path.join(labels_dir, f"{name}.txt")
        # Create empty txt file
        open(label_path, "w").close()

print("✅ Labels folder created with empty txt files!")

import shutil
shutil.move("mydataset/foggy_zurich_direct/test_images", "mydataset/foggy_zurich_direct/val/images")
shutil.move("mydataset/foggy_zurich_direct/test_labels", "mydataset/foggy_zurich_direct/val/labels")

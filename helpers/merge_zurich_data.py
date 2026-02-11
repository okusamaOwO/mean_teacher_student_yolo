import os
import shutil

light = "/workspace/mydataset/foggy_zurich_direct/light_images"
medium = "/workspace/mydataset/foggy_zurich_direct/medium_images"
merged = "/workspace/mydataset/foggy_zurich_direct/images"

# Create merged dir if not exists
os.makedirs(merged, exist_ok=True)

def merge_folders(src, dst):
    for filename in os.listdir(src):
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)

        # If file with same name exists, rename to avoid overwrite
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            new_filename = f"{name}_{counter}{ext}"
            new_dst_path = os.path.join(dst, new_filename)
            while os.path.exists(new_dst_path):
                counter += 1
                new_filename = f"{name}_{counter}{ext}"
                new_dst_path = os.path.join(dst, new_filename)
            dst_path = new_dst_path

        shutil.copy2(src_path, dst_path)

# Merge both folders
merge_folders(light, merged)
merge_folders(medium, merged)

print("✅ Merging done! Files are in:", merged)

import os

def check_labels(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    for img_file in image_files:
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(directory, label_file)
        if not os.path.exists(label_path):
            print(f"Label file missing for image: {img_file}")
        else:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"Label file empty for image: {img_file}")
                else:
                    print(f"Label file for {img_file} contains: {content}")

# Check all directories
check_labels("/Users/admin/Documents/YOLOv8-Fire-detection/data/data_fire/train")

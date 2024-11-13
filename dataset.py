import os
import random
import shutil

# Paths
images_folder = 'images/'  # Folder containing images
labels_folder = 'labels/'  # Folder containing labels
train_images_folder = 'train/images/'
train_labels_folder = 'train/labels/'
valid_images_folder = 'valid/images/'
valid_labels_folder = 'valid/labels/'

# Create directories for train and validation splits
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(valid_images_folder, exist_ok=True)
os.makedirs(valid_labels_folder, exist_ok=True)

# Get list of all image files and corresponding label files
images = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
labels = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in images]  # Assuming labels have same names with .txt extension

# Ensure each image has a corresponding label
all_files = list(zip(images, labels))

# Shuffle the files
random.shuffle(all_files)

# Split into 80-20
split_index = int(0.8 * len(all_files))
train_files = all_files[:split_index]
valid_files = all_files[split_index:]

# Move files to respective folders
def move_files(file_list, image_dest, label_dest):
    for image_file, label_file in file_list:
        # Paths for source and destination
        image_src = os.path.join(images_folder, image_file)
        label_src = os.path.join(labels_folder, label_file)
        image_dst = os.path.join(image_dest, image_file)
        label_dst = os.path.join(label_dest, label_file)
        
        # Check if both image and label exist before moving
        if os.path.exists(image_src) and os.path.exists(label_src):
            shutil.copy(image_src, image_dst)
            shutil.copy(label_src, label_dst)
        else:
            print(f"Missing file for {image_file}, skipping...")

# Move training files
move_files(train_files, train_images_folder, train_labels_folder)

# Move validation files
move_files(valid_files, valid_images_folder, valid_labels_folder)

print("Data split completed successfully!")
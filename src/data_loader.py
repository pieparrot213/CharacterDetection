import os
import cv2

def load_data(data_dir):
    images = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                images.append(cv2.imread(image_path))
                label_path = os.path.join(root.replace('images', 'labels'), file.replace('.jpg', '.txt'))
                try:
                    with open(label_path, 'r') as f:
                        labels.append(f.read().strip())
                except Exception as e:
                    print(f"Error loading label for {file}: {e}")
    return images, labels
# Add print statements to debug
train_images, train_labels = load_data('data/image')  # Replace 'data_dir' with 'data'
print(f"Number of training images: {len(train_images)}")

validate_images, validate_labels = load_data('data')  # Replace 'data_dir' with 'data'
print(f"Number of validation images: {len(validate_images)}")
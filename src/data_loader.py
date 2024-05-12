# data_loader.py
import os
import cv2

os.chdir('D:/Task/2324_II/Image Processing/CharacterDetection')

def load_train_image(train_image_dir):
    train_images = []
    for file in os.listdir(train_image_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(train_image_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                train_images.append(img)
    return train_images


def load_val_image(val_image_dir):
    val_images = []
    for file in os.listdir(val_image_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(val_image_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                val_images.append(img)
    return val_images

def load_train_label(train_label_dir):
    train_label = []
    for file in os.listdir(train_label_dir):
        if file.endswith(".txt"):
            train_label.append(file)
    return train_label

def load_val_label(val_label_dir):
    val_label = []
    for file in os.listdir(val_label_dir):
        if file.endswith(".txt"):
            val_label.append(file)
    return val_label

# train_images = load_train_image('data/images/train')
# validate_images = load_val_image('data/images/validate')
# train_labels = load_train_label('data/labels/train')
# validate_labels = load_val_label('data/labels/validate')
# print(len(train_labels))
# print(len(train_images))
# print(len(validate_labels))
# print(len(validate_images))
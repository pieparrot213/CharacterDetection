# train.py
from data_loader import load_train_label, load_train_image, load_val_image, load_val_label
from model import create_model

# Load data
train_images = load_train_image('data/images/train')
validate_images = load_val_image('data/images/validate')
train_labels = load_train_label('data/labels/train')
validate_labels = load_val_label('data/labels/validate')


# Define model parameters
input_shape = train_images[0].shape
num_classes = len(set(train_labels))

# Create and train model
model = create_model(input_shape, num_classes)
model.fit(train_images, train_labels, epochs=10, validation_data=(validate_images, validate_labels))

from data_loader import load_data
from model import create_model

# Load data
train_images, train_labels = load_data('data')
print(f"Number of training images: {len(train_images)}")

validate_images, validate_labels = load_data('data')
print(f"Number of validation images: {len(validate_images)}")

# Define model parameters
if train_images:
    input_shape = train_images[0].shape
    num_classes = len(set(train_labels))

    # Create and train model
    model = create_model(input_shape, num_classes)
    model.fit(train_images, train_labels, epochs=10, validation_data=(validate_images, validate_labels))
else:
    print("No training images found. Please check your dataset.")

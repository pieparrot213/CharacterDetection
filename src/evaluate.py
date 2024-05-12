from data_loader import load_data
from model import create_model

# Load data
test_images, test_labels = load_data('data/images/test')

# Load model
model = create_model(input_shape, num_classes)
model.load_weights('model/your_trained_model.h5')

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')

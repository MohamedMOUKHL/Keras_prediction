# from flask import Flask, render_template, request
# from PIL import Image
# import numpy as np
# from keras.models import load_model

# import os

# # Get the current working directory
# current_directory = os.getcwd()

# # Define the file path to the model
# #"C:\MOUKHLISSI\GLD\2A\Application AI\Computer_vision_Keras\cifar10_model.h5"
# #model_file_path = "C:\\MOUKHLISSI\\GLD\\2A\\Application AI\\Computer_vision_Keras\\cifar10_model.h5"
# #model_file_path = os.path.join(current_directory, "C:\MOUKHLISSI\GLD\2A\Application AI\Computer_vision_Keras\cifar10_model.h5")


# # Define the file path to the model
# model_file_path = os.path.join(current_directory, 'ciraf10_model.h5')

# # Check if the file exists
# if os.path.isfile(model_file_path):
#     # Load the model using the correct file path
#     model = load_model(model_file_path)
# else:
#     print(f"Model file {model_file_path} does not exist.")


# # Load the model using the correct file path
# model = load_model(model_file_path)


# app = Flask(__name__)

# # Load the trained model
# model = load_model('ciraf10_model.h5')  # Replace 'your_model.h5' with the name of your downloaded model file

# # Define class labels
# class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# # Preprocess image
# def preprocess_image(image):
#     image = image.convert("RGB")  # Convert to RGB mode (if not already)
#     image = image.resize((32, 32))  # Resize image to match model input shape
#     image_array = np.array(image)
#     image_array = image_array.astype('float32') / 255.0  # Normalize pixel values
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get uploaded image
#     uploaded_file = request.files['file']
#     image = Image.open(uploaded_file)

#     # Preprocess image
#     processed_image = preprocess_image(image)

#     # Make prediction
#     prediction = model.predict(processed_image)
#     predicted_class = class_labels[np.argmax(prediction)]

#     return render_template('index.html', prediction=predicted_class, image_file=uploaded_file)

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model

import os

# Get the current working directory
current_directory = os.getcwd()

# Define the file path to the model
model_file_path = os.path.join(current_directory, 'cifar10_model.h5')

# Check if the file exists
if os.path.isfile(model_file_path):
    # Load the model using the correct file path
    model = load_model(model_file_path)
else:
    print(f"Model file {model_file_path} does not exist.")
    exit()  # Exit the program if the model file does not exist

app = Flask(__name__)

# Define class labels
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB mode (if not already)
    image = image.resize((32, 32))  # Resize image to match model input shape
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image
    uploaded_file = request.files['file']
    image = Image.open(uploaded_file)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_class, image_file=uploaded_file)

if __name__ == '__main__':
    app.run(debug=True)
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from tensorflow.keras.models import load_model as load_tf_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the brain tumor detection model
model_brain_tumor = load_model('BrainTumor10Epochs.h5')

# Load the MRI classification model
model_mri = load_tf_model('MriorNot.keras')

print('Models loaded. Check http://127.0.0.1:5000/')


def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Brain Tumor Detected"


def is_mri_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict using the MRI classification model
    result = model_mri.predict(x)
    return result[0][0] < 0.5  # Assuming it returns probability of being MRI


def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    # Predict using the brain tumor detection model if it's MRI
    result = (model_brain_tumor.predict(input_img) > 0.5).astype("int32")
    return result[0][0]  # Assuming binary classification

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Check if the uploaded image is an MRI scan
        if is_mri_image(file_path):
            # If MRI, predict brain tumor
            value = get_result(file_path)
            result = get_class_name(value)
            return result
        else:
            return "Uploaded image is not an MRI scan."

    return "Invalid request"

if __name__ == '__main__':
    app.run(debug=True)
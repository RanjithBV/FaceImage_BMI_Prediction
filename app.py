from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl','rb'))

def prepare_image(image, target):
    # Resize the image to the target dimensions
    image = cv2.resize(image, target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = prepare_image(image, target=(224, 224))

    prediction = model.predict(image)[0][0]
    prediction = float(prediction)
    return jsonify({'BMI': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
from flask import Flask, render_template, request, redirect
import os
import pickle
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load the pickled model
with open('cat_dog.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function — change according to your model’s requirements
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # 👈 Change if your model expects a different size
    img = img.convert('RGB')      # Ensure it's RGB
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 256, 256, 3)  # 👈 Adjust shape for CNN input
    return img_array

@app.route('/')
def index():
    return render_template('index.html', show_image=False)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    if file:
        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        file.save(filepath)

        img = preprocess_image(filepath)
        prediction = model.predict(img)

        # Handle output (change based on model output type)
        if isinstance(prediction, np.ndarray):
            pred_value = int(np.round(prediction[0]))  # e.g., [1] or [0]
        else:
            pred_value = int(prediction)

        label = 'Dog 🐶' if pred_value == 1 else 'Cat 🐱'
        return render_template('index.html', prediction=label, show_image=True)


if __name__ == '__main__':
    app.run(debug=True)
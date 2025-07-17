from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
model = load_model('Training/rice.h5')
class_names = sorted(os.listdir('Data/train'))

# In-memory prediction history
prediction_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            img_path = os.path.join('static', img_file.filename)
            img_file.save(img_path)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            # Store prediction in history
            prediction_history.append({
                "filename": img_file.filename,
                "predicted_class": predicted_class,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return render_template('results.html', prediction=predicted_class, img_path=img_file.filename)
    return render_template('index.html')

@app.route('/plots')
def plots():
    # Render a template that shows the plots
    return render_template('plots.html')

@app.route('/history')
def history():
    # Render a template that shows the prediction history
    return render_template('history.html', history=prediction_history)

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True)
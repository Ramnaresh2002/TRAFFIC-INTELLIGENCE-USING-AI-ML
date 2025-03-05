import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='C:/Users/Ram Naresh/Desktop/Python/Flask/template')

# Load model and scaler
base_path = r"C:/Users/Ram Naresh/Desktop/Python/"
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
scale = pickle.load(open(os.path.join(base_path, "encoder.pkl"), "rb"))

@app.route('/')  # Home page
def home():
    return render_template('index2.html')

@app.route('/predict', methods=["POST", "GET"])  # Prediction route
def predict():
    try:
        # Read input features
        input_feature = [float(x) for x in request.form.values()]
        features_values = np.array([input_feature])

        # Define column names
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
                 'hours', 'minutes', 'seconds']

        # Convert to DataFrame
        data = pd.DataFrame(features_values, columns=names)

        # Apply scaling (use transform instead of fit_transform)
        data = scale.transform(data)

        # Make prediction
        prediction = model.predict(data)

        # Display result
        text = f"Estimated Traffic Volume is: {prediction[0]}"
        return render_template("index.html", prediction_text=text)
            
    except Exception as e:
           return render_template("index.html", prediction_text=f"Error: {str(e)}")

    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)

import os
import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load your trained model (update path if needed)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'random_forest_model_top10.joblib')
model = joblib.load(model_path)

# Features your model expects
selected_features = [
    'google_index', 'page_rank', 'web_traffic', 'nb_hyperlinks', 'nb_www',
    'domain_age', 'phish_hints', 'safe_anchor', 'ratio_extRedirection', 'ratio_digits_url'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract features from form
            feature_values = [float(request.form[feature]) for feature in selected_features]
            input_df = pd.DataFrame([feature_values], columns=selected_features)

            # Make prediction
            prediction = model.predict(input_df)[0]
            result = "Phishing" if prediction == 1 else "Legitimate"

            return render_template('result.html', prediction=result)
        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('index.html', selected_features=selected_features)

if __name__ == '__main__':
    app.run(debug=True)

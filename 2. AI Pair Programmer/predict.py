from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('titanic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract features from form data
    features = np.array([
        int(data['pclass']),
        int(data['sex'] == 'male'),
        float(data['age']),
        int(data['sibsp']),
        int(data['parch']),
        float(data['fare']),
        int(data['embarked'] == 'Q'),
        int(data['embarked'] == 'S')
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]

    # Return result
    return jsonify({'survival_prediction': int(prediction)})

if __name__ == '__main__':
    # Load the pre-trained model
    model = joblib.load('titanic_model.pkl')
    print("Model loaded successfully.")
    
    # Run the Flask app
    app.run(debug=True)
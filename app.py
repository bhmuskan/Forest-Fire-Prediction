from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        oxygen = float(request.form['oxygen'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        input_data = np.array([[oxygen, temperature, humidity]], dtype=np.float32)
        prediction = model.predict(input_data)[0]

        result = '🔥 Fire Occurrence: Yes' if prediction == 1 else '✅ Fire Occurrence: No'
        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

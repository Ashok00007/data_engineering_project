import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Pickle_RL_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    print(prediction)
    output =prediction[0]

    return render_template('index.html', prediction_text='Prediction of item sales {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

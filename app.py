import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("disaster-tweets.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    prediction = model.predict([data['paragraph']])  # Removed np.array()

    output = int(prediction[0])  # Convert numpy.int64 to int

    return jsonify(output)

if __name__ == '__main__':
    flask_app.run(port=5000, debug=True)
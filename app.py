import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
with open('iris.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        
        prediction = model.predict(data)[0]
        
        species = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        result = species[prediction]
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

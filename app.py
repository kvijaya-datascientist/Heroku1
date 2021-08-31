import numpy as np
import pandas as pd
import pickle
from flask import Flask,render_template,request,jsonify

salPred = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@salPred.route('/')
def home():
    return render_template('index.html')

@salPred.route('/predict', methods=['POST'])
def predict():
    '''for rendering results on HTML GUI'''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Employee salary should be $ {}'.format(output))

if __name__ == "__main__":
    salPred.run(debug=True)
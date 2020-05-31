# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:12:47 2020

@author: Sumit Keshav
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rf_all.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    int_features = [float(x) for x in request.form.values()]
    arr = np.array(int_features)
    p1 = arr[:7]
    p2 = arr[7:8]
    p3 = arr[8:11]
    p4 = arr[11:]
    c = np.concatenate((p1,p3)).reshape(1,-1)
    std = pickle.load(open("scaler.pkl","rb"))
    res = std.transform(c).squeeze()
    f_arr = np.concatenate((res[:7],p2,res[7:],p4))              
    
    final_features = f_arr.reshape(1,-1)
    
    prediction = model.predict_proba(final_features)
    
    output = '{0:.{1}f}'.format(prediction[0][1],2)
    
    if output>str(0.46):
        return render_template('index2.html',pred = 'Warning: SKU might be OUT OF STOCK.\nProbability of OOS occuring is :{}'.format(output))
    else:
        return render_template('index2.html',pred = 'SKU is IN STOCK.\nProbability of OOS occuring is: {}'.format(output))
        
        
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, flash
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


app = Flask(__name__, template_folder='template')
app.secret_key = 'O.\x89\xcc\xa0>\x96\xf7\x871\xa2\xe6\x9a\xe4\x14\x91\x0e\xe5)\xd9'
classifier=pickle.load(open('svm_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("Diabetes_Prediction.html")

def std_scalar(df):
    std_X = StandardScaler()
    x =  pd.DataFrame(std_X.fit_transform(df))
    return x

def pipeline(features):
    steps = [('scaler', StandardScaler()), ('SVM', svm_model)]
    pipe = Pipeline(steps)
    return pipe.fit_transform(features)

@app.route('/send', methods=['POST'])
def getdata():
    if request.method == 'POST':
        try:
            preg = int(request.form['Pregnancies'])
            glucose = int(request.form['Glucose'])
            bp = int(request.form['BloodPressure'])
            st = int(request.form['SkinThickness'])
            insulin = int(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            dpf = float(request.form['DiabetesPedigreeFunction'])
            age = int(request.form['Age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('Result.html', prediction=my_prediction)
        except ValueError:
            flash(
                'Invalid input. Please fill in the form with appropriate values', 'info')
            return "Invalid input. Please fill in the form with appropriate values"


if __name__=="__main__":
    app.run(debug=True)

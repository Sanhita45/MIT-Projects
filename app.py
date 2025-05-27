import pickle
import pandas as pd
from flask import Flask, render_template, request

app= Flask(__name__)
model=pickle.load(open('model1.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    if request.method == 'POST':
        # Retrieve form data
        age = int(request.form['age'])
        anaemia = int(bool(request.form.get('anaemia')))
        creatinine_phosphokinase = int(request.form['creatinine_phosphokinase'])
        diabetes = int(bool(request.form.get('diabetes')))
        ejection_fraction = int(request.form['ejection_fraction'])
        high_blood_pressure = int(bool(request.form.get('high_blood_pressure')))
        platelets = int(request.form['platelets'])
        serum_creatinine = float(request.form['serum_creatinine'])
        serum_sodium = int(request.form['serum_sodium'])
        sex = int(bool(request.form['sex'] == 'male'))  # Convert sex to 0 or 1
        smoking = int(bool(request.form.get('smoking')))

        # Create a dictionary with the received data
        data_dict = {
            'age': [age],
            'anaemia': [anaemia],
            'creatinine_phosphokinase': [creatinine_phosphokinase],
            'diabetes': [diabetes],
            'ejection_fraction': [ejection_fraction],
            'high_blood_pressure': [high_blood_pressure],
            'platelets': [platelets],
            'serum_creatinine': [serum_creatinine],
            'serum_sodium': [serum_sodium],
            'sex': [sex],
            'smoking': [smoking],
        }

    # Create a DataFrame from the dictionary
        input_data = pd.DataFrame(data_dict)
        input_data= scaler.transform(input_data)
        prediction = (model.predict(input_data)>0.5).astype(int).flatten()
        if prediction == [1]:
            return "You may have heart attack in future"
        else:
            return "You are healthy"

if __name__== '__main__':
    app.run(host='0.0.0.0',port=8080)

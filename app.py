from flask import Flask, render_template, request
import numpy as np
import pickle

model=pickle.load(open('HeartAttackPrediction.pkl','rb'))

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def new():
    return render_template('heart_attack.html')

@app.route('/heart-attack-prediction', methods=['POST', 'GET'])
def predict():
  data1=float(request.form['age'])
  data2=float(request.form['sex'])
  data3=float(request.form['chest-pain-type'])
  data4=float(request.form['rbp'])
  data5=float(request.form['chol'])
  data6=float(request.form['fbs'])
  data7=float(request.form['rest_ecg'])
  data8=float(request.form['max-heart-rate'])
  data9=float(request.form['exang'])
  data10=float(request.form['oldpeak'])
  data11=float(request.form['slp'])
  data12=float(request.form['ca'])
  data13=float(request.form['thal'])
  features=np.array([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10, data11, data12, data13])
  pred = model.predict([features])
  print(pred)
  def final_statement():
    if pred == 1:
      return 'Result: The model has predicted that you will not suffer from heart attack. However, you should take care of your heart health.'
    elif pred == 0:
      return 'Result: This model has predicted that you have high chances of suffering from heart attack. You should immediately consult a doctor.'
  return render_template('heart_attack.html',statement=final_statement())

app.run(port=4999, debug=['True'])
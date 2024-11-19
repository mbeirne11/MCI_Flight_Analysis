#imports
from bson import json_util
from flask import Flask, render_template , request,json
import pandas as pd
import tensorflow as tf
#create flask
app = Flask(__name__)

years_data = pd.read_csv('MCI-Flight-Analysis/Resources/years_data.csv')
months_data = pd.read_csv('MCI-Flight-Analysis/Resources/months_data.csv')
dows_data = pd.read_csv('MCI-Flight-Analysis/Resources\dows_data.csv')
carriers_data = pd.read_csv('MCI-Flight-Analysis/Resources/airlines_data.csv')
X = pd.read_csv('MCI-Flight-Analysis/Resources/X.csv')

#home route
@app.route("/")
def default():
    return render_template('py_index.html')
#get data from mongo db route
@app.route("/get_data")
def get_data():
    return json_util.dumps(X)
@app.route("/get_Month_data")
def get_months_data():
    return json_util.dumps(months_data)
@app.route("/get_Year_data")
def get_years_data():
    return json_util.dumps(years_data)
@app.route("/get_Day of the Week_data")
def get_dows_data():
    return json_util.dumps(dows_data)
@app.route("/get_Airline_data")
def get_carriers_data():
    return json_util.dumps(carriers_data)

model = tf.keras.models.load_model('MCI-Flight-Analysis/Resources/nn_model.keras')
# predict route that will post to the home route
@app.route('/predict', methods=['POST'])
def predict():
    #grab data used in payload in model.js
    if request.method == 'POST':
        data = request.form['data']
        #convert the data to json
        data = json.loads(data)
        #predict both models
        prediction = model.predict([data['data']])
        print(prediction)
        #reset the status
        status = ''
        #update status
        if (prediction>=.5):
            status = 'Delayed'
        else :
            status = 'On Time' 
        return f'Predicted Status: {status}, Prediction: {"{:.2f}".format(prediction[0][0])}'

if __name__ == '__main__':
    app.run(debug=True, port=5500)

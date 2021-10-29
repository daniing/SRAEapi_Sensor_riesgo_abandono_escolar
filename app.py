import csv
import numpy as np
import pandas as pd
import pickle
import joblib
import math
import datetime as dt 
from datetime import datetime, date, time, timedelta
from flask import Flask, request, jsonify, render_template
#from flask_restful import Resource, Api
from sklearn.preprocessing import StandardScaler

# Create flask app
app = Flask(__name__)

model_MLP = pickle.load(open('model_MLP.pkl','rb'))
    
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    
    fech_n_ = request.form.get('edad')
    
    fech1 = datetime.strptime(fech_n_, '%Y-%m-%d').date()
    fech2 = date.today()# criterio de medición (año) 
    edad_ = round((fech2-fech1)/dt.timedelta(365,5,49,12),3) #promedio de años comunes y bisiestos 
    
    feature_hc = request.form.get('hermanos_Colegio')
    feature_g = request.form.get('gender')
    feature_grado_a = request.form.get('curso')
    feature_score_sisben = request.form.get('puntaje_sisben')
    feature_distance = request.form.get('distancia')
    feature_year_in = request.form.get('year_in')
    feature_grado_in = request.form.get('grado_in')
    feature_math = request.form.get('math')
    feature_cn = request.form.get('cn')
    feature_ing = request.form.get('ing')
    feature_esp = request.form.get('esp')
    feature_catp = request.form.get('cp')
    feature_soc = request.form.get('soc')
    feature_rel = request.form.get('rel')
    feature_art = request.form.get('art')
    feature_edf = request.form.get('edf')
    feature_tec = request.form.get('tech')
    feature_emp = request.form.get('emp')
    feature_etv = request.form.get('etv')
    
    list_fini = [feature_hc,feature_g,feature_grado_a,feature_score_sisben,
                 feature_distance,feature_year_in,feature_grado_in,
                 feature_math,feature_cn,feature_ing,feature_esp,
                 feature_catp,feature_soc,feature_rel,feature_art,
                 feature_edf,feature_tec,feature_emp,feature_etv]
    
    prediction_lf = model_MLP.predict_proba(late_test)[:,1]
    return render_template("index.html", prediction_text = "El riesgo de abandono es de {}".format(list_fini))

if __name__ == "__main__":
    app.run(debug=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:22:33 2021

@author: danielzapatamedina
"""

import csv
import numpy as np
import pandas as pd
import pickle
import joblib
import math
import scipy.stats as stats
import datetime as dt 
from datetime import datetime, date, time, timedelta
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from sklearn.preprocessing import StandardScaler

# Create flask app
flask_app = Flask(__name__)
#model = pickle.load(open("model_api_prueba.pkl", "rb"))

filename_model_d = 'model_api_demog.pkl'
filename_model_md = 'model_api_md.pkl'
filename_model_a = 'model_api_a.pkl'
filename_model_ma = 'model_api_ma.pkl'
filename_latefusion = 'model_latefusion.pkl'

model_d = joblib.load(filename_model_d)
model_md = joblib.load(filename_model_md)
model_a = joblib.load(filename_model_a)
model_ma = joblib.load(filename_model_ma)
model_lf = joblib.load(filename_latefusion)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    features_ini = [str(x) for x in request.form.values()]
    features_ini_ = np.array(features_ini)
    features = np.array(features_ini_)
    features_float = features[1::]
    features_float_ = [float(x) for x in features_float]
    feature_date = features[:1:]
    #fech_n = request.form.get('fecha_nacimiento')
    fech_n_ = " ".join(str(x) for x in feature_date)
    
    fech1 = datetime.strptime(fech_n_, '%Y-%m-%d').date()
    fech2 = date.today()# criterio de medición (año) 
    edad_ = round((fech2-fech1)/dt.timedelta(365,5,49,12),3) #promedio de años comunes y bisiestos 
    
    feature_hc = features[1].astype(float)
    feature_g = features[2].astype(float)
    feature_grado_a = features[3].astype(float)
    feature_score_sisben = features[4].astype(float)
    feature_distance = features[5].astype(float)
    feature_year_in = features[6].astype(float)
    feature_grado_in = features[7].astype(float)
    feature_math = features[8].astype(float)
    feature_cn = features[9].astype(float)
    feature_ing = features[10].astype(float)
    feature_esp = features[11].astype(float)
    feature_catp = features[12].astype(float)
    feature_soc = features[13].astype(float)
    feature_rel = features[14].astype(float)
    feature_art = features[15].astype(float)
    feature_edf = features[16].astype(float)
    feature_tec = features[17].astype(float)
    feature_emp = features[18].astype(float)
    feature_etv = features[19].astype(float)
    
    hc_ = feature_hc
    
    m1_d = edad_ - feature_grado_a + 5
    
    m2_d = np.where((feature_grado_a==feature_grado_in),0,
                     np.where((((2019-feature_year_in)+feature_year_in)-feature_grado_in)<0,
                     0,((2019-feature_year_in)+feature_year_in)-feature_grado_in))
    
    mCM = (feature_cn+feature_math)/2
    mSC = (feature_soc+feature_catp)/2
    mCh = (feature_esp+feature_ing)/2
    mEr = (feature_etv+feature_rel)/2
    mAt = (feature_art+feature_tec)/2
    
    mah=(feature_cn+feature_math+feature_soc+feature_esp+feature_ing)/5
    mbh=(feature_tec+feature_rel+feature_catp+feature_art+feature_etv+feature_emp+feature_edf)/7
    
    m1_a = (feature_cn/5)*((mah+mbh)/2)
    m2_a = (feature_emp/5)*((mah+mbh)/2)
    m3_a = (feature_math/5)*((mah+mbh)/2)
    m4_a = (feature_edf/5)*((mah+mbh)/2)
    
    list_fmd = [hc_,m1_d,m2_d]
    list_fma = [mCM,mSC,mCh,mEr,mAt,mah,mbh,m1_a,m2_a,m3_a,m4_a]
    
    features_d = [np.array(features_ini[1:8])]
    features_d_ = [np.append(features_d, edad_)]
    features_md = [np.array(list_fmd)]
    features_a = [np.array(features_ini[8:20])]
    features_ma = [np.array(list_fma)]
    
    
    prediction_d = model_d.predict_proba(features_d_)
    
    prediction_md = model_md.predict_proba(features_md)
    
    prediction_a = model_a.predict_proba(features_a)
    
    prediction_ma = model_ma.predict_proba(features_ma)
    
    late_test_dam =  np.hstack((prediction_d,prediction_a,prediction_md,prediction_ma))


    # Entrena y evalúa el clasificador final a partir de la representación late fusion
    stdSlr = StandardScaler().fit(late_test_dam)
    late_test =  stdSlr.transform(late_test_dam)
    
    prediction_lf = model_lf.predict_proba(late_test)[0]
    
    return render_template("index.html", prediction_text = "El riesgo de abandono es de {}".format(prediction_lf))

if __name__ == "__main__":
    flask_app.run(debug=True)
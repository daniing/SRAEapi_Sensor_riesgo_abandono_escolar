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

#Call model
model_api_demog = pickle.load(open('model_api_demog.pkl','rb'))
model_api_md = pickle.load(open('model_api_md.pkl','rb'))
model_api_a = pickle.load(open('model_api_a.pkl','rb'))
model_api_ma = pickle.load(open('model_api_ma.pkl','rb'))
model_latefusion = pickle.load(open('model_latefusion.pkl','rb'))


@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    fech_n_ = request.form.get('edad')
    fech1 = datetime.strptime(fech_n_, '%Y-%m-%d').date()
    fech2 = date.today()# criterio de medición (año) 
    edad_ = round((fech2-fech1)/dt.timedelta(365,5,49,12),3) #promedio de años comunes y bisiestos 
    
    feature_hc = float(request.form.get('hermanos_Colegio'))
    feature_g = float(request.form.get('gender'))
    feature_grado_a = float(request.form.get('curso'))
    feature_score_sisben = float(request.form.get('puntaje_sisben'))
    feature_distance = float(request.form.get('distancia'))
    feature_year_in = float(request.form.get('year_in'))
    feature_grado_in = float(request.form.get('grado_in'))
    feature_math = float(request.form.get('math'))
    feature_cn = float(request.form.get('cn'))
    feature_ing = float(request.form.get('ing'))
    feature_esp = float(request.form.get('esp'))
    feature_catp = float(request.form.get('cp'))
    feature_soc = float(request.form.get('soc'))
    feature_rel = float(request.form.get('rel'))
    feature_art = float(request.form.get('art'))
    feature_edf = float(request.form.get('edf'))
    feature_tec = float(request.form.get('tech'))
    feature_emp = float(request.form.get('emp'))
    feature_etv = float(request.form.get('etv'))
    
    
    list_demog = [edad_,feature_hc,feature_g,feature_score_sisben,feature_distance,feature_grado_in,
                 feature_year_in,feature_grado_a]
    
    list_acad= [feature_math,feature_cn,feature_ing,feature_esp,
                 feature_catp,feature_soc,feature_rel,feature_art,
                 feature_edf,feature_tec,feature_emp,feature_etv]
    
    #features_array = np.array(features_float_)
    
    m1_d = edad_ - feature_grado_a + 5
    
    #m2_d = np.where((feature_grado_a==feature_grado_in),0,
    #                 np.where((2019-feature_year_in+feature_grado_in)-feature_grado_a)<0,
    #                 0,((2019-feature_year_in+feature_grado_in)-feature_grado_a))
    
    m2_d = abs((2019-feature_year_in+feature_grado_in)-feature_grado_a)
    
    m3_d = 0.415*feature_g+ 0.089*edad_ - 0.009*feature_score_sisben-3.10
    
    m4_d = 0.412*feature_g+ 0.085*feature_hc- 0.009*feature_score_sisben-1.77
    
    m5_d = 0.399*feature_g+ 0.084*edad_+ 0.161*feature_distance-3.25
                                         
    m6_d = 0.397*feature_g+ 0.065*feature_hc+ 0.168*feature_distance-1.99


    list_md = [m1_d,m2_d,m3_d,m4_d,m5_d,m6_d]
    
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
    
    
    list_ma = [mCM,mSC,mCh,mEr,mAt,mah,mbh,m1_a,m2_a,m3_a,m4_a]
    
    
    features_d = [np.array(list_demog)]
    features_md = [np.array(list_md)]
    features_a = [np.array(list_acad)]
    features_ma = [np.array(list_ma)]
    
    # Entrena y evalúa el clasificador para características demográficas
    prediction_d = model_api_demog.predict_proba(features_d)

    # Entrena y evalúa el clasificador para métricas de características demográficas
    prediction_md = model_api_md.predict_proba(features_md)
    
    
    # Entrena y evalúa el clasificador para características académicas
    prediction_a = model_api_a.predict_proba(features_a)
    
    # Entrena y evalúa el clasificador para métricas de características académicas
    prediction_ma = model_api_ma.predict_proba(features_ma)
    
    features_latefusion =  np.hstack((prediction_d,prediction_a,prediction_md,prediction_ma))

    # Entrena y evalúa el clasificador final a partir de la representación late fusion
    prediction_lf = model_latefusion.predict_proba(features_latefusion)[:,1][0]
    
    result = round(prediction_lf)
    return render_template("index.html", prediction_text = "El riesgo de abandono es de {} %".format(result))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    #flask_app.config['TEMPLATES_AUTO_RELOAD'] = True

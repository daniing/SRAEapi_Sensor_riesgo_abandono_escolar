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
    
    #hc_ = feature_hc
    features_float_ = [float(x) for x in list_fini]
    
    features_array = np.array(features_float_)
    
    m1_d = edad_ - features_array[2] + 5
    
    m2_d = np.where((features_array[2]==features_array[6]),0,
                     np.where((((2019-features_array[5])+features_array[5])-features_array[6])<0,
                     0,((2019-features_array[5])+features_array[5])-features_array[6]))
    
    mCM = (features_array[8]+features_array[7])/2
    mSC = (features_array[12]+features_array[11])/2
    mCh = (features_array[10]+features_array[9])/2
    mEr = (features_array[18]+features_array[13])/2
    mAt = (features_array[14]+features_array[16])/2
    
    mah=(features_array[8]+features_array[7]+features_array[12]+features_array[10]+features_array[9])/5
    mbh=(features_array[11]+features_array[13]+features_array[14]+features_array[15]+features_array[16]+features_array[17]+features_array[18])/7
    
    m1_a = (features_array[8]/5)*((mah+mbh)/2)
    m2_a = (features_array[17]/5)*((mah+mbh)/2)
    m3_a = (features_array[7]/5)*((mah+mbh)/2)
    m4_a = (features_array[15]/5)*((mah+mbh)/2)
    
    list_fmd = [m1_d,m2_d,mCM,mSC,mCh,mEr,mAt,mah,mbh,m1_a,m2_a,m3_a,m4_a]
    
    features_d = [edad_]+features_float_+list_fmd
    
    #features_d_ = [np.append(features_d, edad_)]
    features_d_ = [np.array(features_d)]
    
    # Entrena y evalúa el clasificador final a partir de la representación late fusion
    #stdSlr = StandardScaler().fit(features_d_)
    #late_test =  stdSlr.transform(features_d_)
    
    prediction_lf = model_MLP.predict_proba(features_d_)[:,1]
    
    return render_template("index.html", prediction_text = "El riesgo de abandono es de {}".format(prediction_lf))

if __name__ == "__main__":
    app.run(debug=True)

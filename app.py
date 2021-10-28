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

#filename_model_d = 'model_api_demog.pkl'
#filename_model_md = 'model_api_md.pkl'
#filename_model_a = 'model_api_a.pkl'
#filename_model_ma = 'model_api_ma.pkl'
#filename_latefusion = 'model_latefusion.pkl'
#filename_MLP = 'model_MLP.pkl'
#model_d = joblib.load(filename_model_d)
#model_md = joblib.load(filename_model_md)
#model_a = joblib.load(filename_model_a)
#model_ma = joblib.load(filename_model_ma)
#model_lf = joblib.load(filename_latefusion)
#model_MLP = joblib.load(filename_MLP)

model_MLP = pickle.load(open('model_MLP.pkl','rb'))

#with open(filename_MLP, 'rb') as f: 
#    model_MLP = pickle.loads(f.read())
    
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    
    #features_ini = [str(x) for x in request.form.values()]
    #features_ini_ = np.array(features_ini)
    #features = np.array(features_ini)
    #features_float = features[1::]
    #features_float_ = [float(x) for x in features_float]
    #feature_date = features[:1:]
    #fech_n = request.form.get('fecha_nacimiento')
    #fech_n_ = " ".join(str(x) for x in feature_date)
    fech_n_ = request.form['edad']
    
    fech1 = datetime.strptime(fech_n_, '%Y-%m-%d').date()
    fech2 = date.today()# criterio de medición (año) 
    edad_ = round((fech2-fech1)/dt.timedelta(365,5,49,12),3) #promedio de años comunes y bisiestos 
    
    feature_hc = request.form['hermanos_Colegio']
    feature_g = request.form['gender']
    feature_grado_a = request.form['curso']
    feature_score_sisben = request.form['puntaje_sisben']
    feature_distance = request.form['distancia']
    feature_year_in = request.form['year_in']
    feature_grado_in = request.form['grado_in']
    feature_math = request.form['math']
    feature_cn = request.form['cn']
    feature_ing = request.form['ing']
    feature_esp = request.form['esp']
    feature_catp = request.form['cp']
    feature_soc = request.form['soc']
    feature_rel = request.form['rel']
    feature_art = request.form['art']
    feature_edf = request.form['edf']
    feature_tec = request.form['tech']
    feature_emp = request.form['emp']
    feature_etv = request.form['etv']
    
    
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
    features_d_ = np.array([features_d])
    
    # Entrena y evalúa el clasificador final a partir de la representación late fusion
    stdSlr = StandardScaler().fit(features_d_)
    late_test =  stdSlr.transform(features_d_)
    
    prediction_lf = model_MLP.predict_proba(late_test)[:,1]
    return render_template("index.html", prediction_text = "El riesgo de abandono es de {}".format(prediction_lf))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

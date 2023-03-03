# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from werkzeug import secure_filename
import os
import datetime
import time

import pandas as pd 
from pandas import Series
import pickle
import numpy
from sklearn.externals import joblib
import io
import base64

import plotly.graph_objects as go

from shap import TreeExplainer, Explanation

# importe nos fonctions
from process_utils import *

# instance flask
app = Flask(__name__)

#pour le rendre joli
Bootstrap(app)

db = SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

my_csv_path = "C:/Users/elie1/Desktop/Diane/OpenClassrooms/Projet 7/App_DS/flask_app/feature_engineered_data_subset.csv"
model_path = "best_model_balanced.pkl"

data = pd.read_csv(my_csv_path, encoding='latin1', index_col=0)
best_model_balanced = pickle.load(open(model_path, 'rb'))

#API
@app.route('/', methods=['GET'  ,'POST'])
def home():

	cust_info = ''
	if request.method == 'POST' and 'cust_info_message' in request.form:
		#on prend le message que rentre l'utilisateur
		cust_ID = request.form['cust_info_message']
		print("POST MSG :   ======"     , cust_ID)
		cust_info = client_info(data, cust_ID)
		if not isinstance(cust_info, str):
			cust_info = cust_info.to_html()


	gauge = ''
	if request.method == 'POST' and 'gauge_message' in request.form:
		#on prend le message que rentre l'utilisateur
		gauge_message = request.form['gauge_message']
		print("POST MSG :   ======"     ,gauge_message)
		# fonctione jauge
		gauge = gauge_from_id(df = data, model = best_model_balanced, cust_ID=gauge_message)
		if not isinstance(gauge, str):
			gauge = gauge.to_html()


	menu_num = dropdown_list(data, 'numerical')
	menu_cat = dropdown_list(data, 'categorical')

	encoded_distrib = ''
	if request.method == 'POST' and 'distrib_message' in request.form:
		# le message que rentre l'utilisateur
		distrib_message = request.form['distrib_message']
		distrib_message2 = request.form['distrib_message2']
		print("POST MSG :   ======"     ,distrib_message)
		print("POST MSG :   ======"     ,distrib_message2)
		plt.clf()
		fig2 = distribution_density_plot(data, cust_ID=distrib_message, num_var=distrib_message2)
		if not isinstance(fig2, str):
			tmpfile = io.BytesIO()
			fig2.savefig(tmpfile, format='png',bbox_inches='tight')
			encoded_distrib = base64.b64encode(tmpfile.getvalue()).decode()


	anova = ''
	if request.method == 'POST' and 'anova_num' in request.form:
		# le message que rentre l'utilisateur
		anova_message = request.form['anova_num']
		anova_message2 = request.form['anova_cat']
		numerical_var = anova_message
		categorical_var = anova_message2
		print("POST MSG :   ======"     ,numerical_var)
		print("POST MSG :   ======"     ,categorical_var)
		plt.clf()
		fig3 = anova_boxplot(data, numerical_var, categorical_var)
		if not isinstance(fig3, str):
			tmpfile = io.BytesIO()
			fig3.savefig(tmpfile, format='png',bbox_inches='tight')
			anova = base64.b64encode(tmpfile.getvalue()).decode()


	analyse_bi_num = ''
	if request.method == 'POST' and 'analyse_bi_num1' in request.form:
		# le message que rentre l'utilisateur
		analyse_bi_ID = request.form['analyse_bi_ID']
		analyse_bi_num1 = request.form['analyse_bi_num1']
		analyse_bi_num2 = request.form['analyse_bi_num2']
		print("POST MSG :   ======"     ,analyse_bi_ID)
		print("POST MSG :   ======"     ,analyse_bi_num1)
		print("POST MSG :   ======"     ,analyse_bi_num2)
		plt.clf()
		fig4 = analyse_bivariee_num(data, best_model_balanced, analyse_bi_num1, analyse_bi_num2, analyse_bi_ID)
		if not isinstance(fig4, str):
			tmpfile = io.BytesIO()
			fig4.savefig(tmpfile, format='png',bbox_inches='tight')
			analyse_bi_num = base64.b64encode(tmpfile.getvalue()).decode()

	
	global_shap_graph = ''
	if request.method == 'POST' and 'global_shap_nb' in request.form:
		global_shap_nb = request.form['global_shap_nb']
		plt.clf()
		fig5 = global_shap(data, best_model_balanced, int(global_shap_nb))
		tmpfile = io.BytesIO()
		plt.savefig(tmpfile, format='png',bbox_inches='tight')
		global_shap_graph = base64.b64encode(tmpfile.getvalue()).decode()
	


	local_shap_graph = ''
	if request.method == 'POST' and 'local_shap_ID' in request.form:
		# le message que rentre l'utilisateur
		local_shap_ID = request.form['local_shap_ID']
		print("POST MSG :   ======"     ,local_shap_ID)
		plt.clf()
		fig6 = local_shap(data, best_model_balanced, local_shap_ID)
		if not isinstance(fig6, str):
			tmpfile = io.BytesIO()
			plt.savefig(tmpfile, format = "png", bbox_inches='tight')
			local_shap_graph = base64.b64encode(tmpfile.getvalue()).decode()


	# will put the result at "prediction" and "prediction2" in home.html
	return render_template('home.html',
		prediction0 = cust_info,
		prediction1 = gauge,
		prediction2 = encoded_distrib,
		prediction3 = anova,
		menu_num = menu_num,
		menu_cat = menu_cat,
		prediction4 = analyse_bi_num,
		prediction5 = global_shap_graph,
		prediction6 = local_shap_graph)





if __name__ == '__main__':
	app.run(port = 5001)
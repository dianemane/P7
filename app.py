# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 

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
import matplotlib.pyplot as plt
import lightgbm
import plotly.graph_objects as go

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

my_csv_path = "feature_engineered_data.csv"
model_path = "best_model_balanced.pkl"

data = pd.read_csv(my_csv_path, encoding='latin1', index_col=0)
best_model_balanced = pickle.load(open(model_path, 'rb'))

#API
@app.route('/', methods=['GET'  ,'POST'])
def home():
	gauge = ''
	if request.method == 'POST' and 'gauge_message' in request.form:
		#on prend le message que rentre l'utilisateur
		gauge_message = request.form['gauge_message']
		print("POST MSG :   ======"     ,data_id)
		# fonctione jauge
		gauge = gauge_from_id(df = data, model = best_model_balanced, cust_ID=gauge_message)
		if not isinstance(gauge, str):
			gauge = gauge.to_html()


	encoded_distrib = ''
	if request.method == 'POST' and 'distrib_message' in request.form:
		# le message que rentre l'utilisateur
		distrib_message = request.form['distrib_message']
		distrib_message2 = request.form['distrib_message2']
		num_var = distrib_message2
		print("POST MSG :   ======"     ,data_id)
		print("POST MSG :   ======"     ,num_var)
		fig2 = distribution_density_plot(data, cust_ID=distrib_message, num_var=num_var)
		if not isinstance(fig2, str):
			tmpfile = io.BytesIO()
			fig2.savefig(tmpfile, format='png')
			encoded_distrib = base64.b64encode(tmpfile.getvalue()).decode()


	'''anova = ''
	if request.method == 'POST' and 'anova_message' in request.form:
		# le message que rentre l'utilisateur
		anova_message = request.form['anova_message']
		anova_message2 = request.form['anova_message2']
		numerical_var = anova_message
		categorical_var = anova_message2
		print("POST MSG :   ======"     ,numerical_var)
		print("POST MSG :   ======"     ,categorical_var)
		fig3 = anova_boxplot(data, numerical_var, categorical_var)
		if not isinstance(fig3, str):
			tmpfile = io.BytesIO()
			fig3.savefig(tmpfile, format='png')
			anova = base64.b64encode(tmpfile.getvalue()).decode()'''


	#anova = ''
	#if request.method == 'GET' and 'anova_num' in request.form:
		# le message que rentre l'utilisateur
	menu_num = dropdown_list(data, 'numerical')
	menu_cat = dropdown_list(data, 'categorical')

	anova = ''
	if request.method == 'POST' and 'anova_num' in request.form:
		# le message que rentre l'utilisateur
		anova_message = request.form['anova_num']
		anova_message2 = request.form['anova_cat']
		numerical_var = anova_message
		categorical_var = anova_message2
		print("POST MSG :   ======"     ,numerical_var)
		print("POST MSG :   ======"     ,categorical_var)
		fig3 = anova_boxplot(data, numerical_var, categorical_var)
		if not isinstance(fig3, str):
			tmpfile = io.BytesIO()
			fig3.savefig(tmpfile, format='png')
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
		fig4 = analyse_bivariee_num(data, best_model_balanced, analyse_bi_num1, analyse_bi_num2, analyse_bi_ID)
		if not isinstance(fig4, str):
			tmpfile = io.BytesIO()
			fig4.savefig(tmpfile, format='png')
			analyse_bi_num = base64.b64encode(tmpfile.getvalue()).decode()


	# will put the result at "prediction" and "prediction2" in home.html
	return render_template('home.html',
		prediction = gauge,
		prediction2 = encoded_distrib,
		prediction3 = anova,
		menu_num = menu_num,
		menu_cat = menu_cat,
		prediction4 = analyse_bi_num)


if __name__ == '__main__':
	app.run(port = 5001)
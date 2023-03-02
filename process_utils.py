import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys
import numpy
from werkzeug import secure_filename
import pandas as pd
from pandas import Series
import pickle

import seaborn as sns
import time
import datetime

import shap
from shap import TreeExplainer, Explanation

import plotly.graph_objects as go



def df_to_X_preprocessing(df, cust_ID):
  if cust_ID == None:
    X = df.loc[:, df.columns != "TARGET"]
    return X
  else:
    cust_ID = str(cust_ID)
    if cust_ID.isnumeric():
      X = df.loc[df['SK_ID_CURR'] == int(cust_ID), df.columns != "TARGET"]
      if X.shape[0] == 0:
        X = "Cet identifiant client n'est pas dans la base"
    else:
      X = "Cet identifiant client n'est pas dans la base : il doit être composé de chiffres uniquement"

    return X


# apply_threshold inutile
def apply_threshold(liste_de_proba, threshold):
    liste_de_pred = []
    for proba in liste_de_proba:
        if (proba > threshold):
            liste_de_pred.append(1)
        else:
            liste_de_pred.append(0)
    return liste_de_pred


def final_model(X, model_proba, threshold):
    y_pred_proba = model_proba.predict_proba(X)[:, 1]
    # y_pred = apply_threshold(y_pred_proba, threshold)
    return threshold, y_pred_proba

def delete_outliers(data, col_list):
  
  Q1 = data[col_list].quantile(0.25)
  Q3 = data[col_list].quantile(0.75)
  IQR = Q3 - Q1

  return data[~((data[col_list] < (Q1 - 1.5 * IQR)) |(data[col_list] > (Q3 + 1.5 * IQR))).any(axis=1)]


def cat_num_var_list(data):

  categorical_var_dum_list = [col for col in data.columns if 'TYPE_' in col and not any(x in col for x in ['_MEAN', '_MAX', '_MIN', 'SUM_', '_VAR'])]
  categorical_var_list = [var[:var.index("TYPE_")+4] for var in categorical_var_dum_list]
  categorical_var_set = set(categorical_var_list)
  categorical_var_list = list(categorical_var_set)

  numerical_var_list = [col for col in data.columns if col not in categorical_var_set and not any(x in col for x in ["TYPE_", '_MEAN', '_MAX', '_MIN', '_SUM', '_VAR'])]
  # menu deroulant
  return (numerical_var_list, categorical_var_list)


def dropdown_list(data, var_type):
  numerical_var_list, categorical_var_list = cat_num_var_list(data)
  if var_type == 'numerical':
    menu = numerical_var_list
  else:
    menu = categorical_var_list
  return menu

def final_proba_1(X, model_proba):
  y_pred_proba = model_proba.predict_proba(X)[:,1]
  return y_pred_proba

  ########## graphs

def client_info(data, cust_ID):
  X = df_to_X_preprocessing(data, cust_ID)

  res = X

  if isinstance(res, pd.DataFrame):
    info_cols = ['SK_ID_CURR',
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_GOODS_PRICE',
    'DAYS_BIRTH',
    'CNT_FAM_MEMBERS',
    'EXT_SOURCE_2']

    df1 = data[info_cols].copy(deep=True)
    df1['Age'] = -df1['DAYS_BIRTH']//365
    df1_id = df1.loc[df1['SK_ID_CURR']==100004,[x for x in info_cols if x != "DAYS_BIRTH"]+['Age']]
    df1_id_space = [sub.replace('_', ' ') for sub in df1_id.columns]

    fig = go.Figure(data=[go.Table(
    header=dict(values=list(df1_id_space),
                fill_color='paleturquoise',
                align='left',
                height=40),
    cells=dict(values=df1_id.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
    ]) 

    res = fig
  return res


  def gauge_from_id(df, cust_ID, model, threshold=0.429):
    X = df_to_X_preprocessing(df, cust_ID)

    res = X

    if isinstance(res, pd.DataFrame):

        thr, y_pred = final_model(X, model, threshold)

        fig = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=y_pred[0],
            mode="gauge+number+delta",
            title={'text': "Model prediction client " + str(cust_ID)},
            delta={'reference': thr, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "black"},
                   'steps': [
                       {'range': [0, thr], 'color': "lightgreen"},
                       {'range': [thr, 1], 'color': "lightpink"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': thr}}))

        res = fig
    return res

# distribution and density plot
def distribution_density_plot(data, cust_ID, num_var):
  X = df_to_X_preprocessing(data, cust_ID)
  res = X

  if isinstance(res, pd.DataFrame):
    fig, axs = plt.subplots(figsize=(7, 8), sharex=True, nrows=2)

    df = data.loc[:,['SK_ID_CURR','TARGET'] + [num_var]]
    # delete outliers
    df = delete_outliers(df, [num_var])

    groupbyTarget = df.groupby("TARGET")[num_var]

    for name, group in groupbyTarget:
        group.plot(kind='kde', ax=axs[0], label=name)
        group.hist(alpha=0.4, ax=axs[1], label=name)


    print(df.loc[df['SK_ID_CURR']==int(cust_ID),num_var])
    print(df.loc[df['SK_ID_CURR']==int(cust_ID),num_var].shape)
    value_id = df.loc[df['SK_ID_CURR']==int(cust_ID),num_var].iloc[0]

    for x in range(2):
        axs[x].set_xlim(df[num_var].min(), df[num_var].max())
        axs[x].axvline(x = value_id, color = 'b')
        axs[x].legend(loc='upper left', frameon=True)


    axs[1].text(float(value_id),0,str("%.2f" % float(value_id))+': '+num_var+' of id '+str(cust_ID),rotation=0)
    #axs[0].legend(loc='upper left', frameon=True)
    axs[0].set_title('Density of '+num_var, fontweight ="bold")
    axs[1].set_title('Distribution of '+num_var, fontweight ="bold")

    res = fig

  return res

def undummify(df, prefix_sep="TYPE_"):
  # create dict, key : column name if no "TYPE_", else column name.split("TYPE_")[0],
  # value : bool if contains "TYPE_"
  cols2collapse = {
      item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
  }
  series_list = []
  for col, needs_to_collapse in cols2collapse.items():
      if needs_to_collapse:
          undummified = (
              # filtre sur les colonnes contenant col_TYPE_
              df.filter(like=col+'TYPE_')
              # numéro de la colonne pour laquelle chaque ligne a sa val max (ie 1 dans le cas des dummies)
              .idxmax(axis=1)
              # ne prend que la deuxième partie du split (donc les étiquettes)
              .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
              .rename(col+'TYPE')
          )
          series_list.append(undummified)
      else:
          series_list.append(df[col])
  undummified_df = pd.concat(series_list, axis=1)
  return undummified_df


def anova_boxplot(data, numerical_var, categorical_var):

  # undummify categorical_var
  data1 = data.copy(deep=True)
  selected_dum_cols = [col for col in data1.columns if categorical_var+'_' in col]
  data1[categorical_var] = undummify(data1[selected_dum_cols])

  # data prep
  modalites = data1[categorical_var].unique()
  modalites_ascending=data1.groupby([categorical_var])[numerical_var].mean().sort_values(ascending=True).index

  groupes = []
  for m in modalites_ascending:
      groupes.append(data1[data1[categorical_var]==m][numerical_var])


  # Propriétés graphiques
  medianprops = {'color':"black"}
  meanprops = {'marker':'o', 'markeredgecolor':'black',
              'markerfacecolor':'firebrick'}

  plt.boxplot(groupes, labels=modalites_ascending, showfliers=False, medianprops=medianprops, 
              vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
  plt.title(numerical_var+' by '+categorical_var)
  plt.xlabel(numerical_var)
  plt.ylabel(categorical_var)
  return plt



def analyse_bivariee_num(df, model, var_x, var_y, cust_ID):
  X = df_to_X_preprocessing(df, cust_ID)
  res = X

  if isinstance(res, pd.DataFrame):
    y_pred_proba = final_proba_1(df_to_X_preprocessing(df, None), model)

    df1 = df.loc[:,['SK_ID_CURR',var_x,var_y]].copy(deep=True)
    df1['y_pred_proba'] = y_pred_proba

    # all rows
    # remove outliers
    df1 = delete_outliers(df1, [var_x,var_y])
    fig, ax = plt.subplots()

    plt.scatter(df1[var_x],df1[var_y],c=df1['y_pred_proba'],cmap='Greens',s=5)
    plt.colorbar()
    plt.xlabel(var_x)
    plt.ylabel(var_y)

    # individual
    df1_id = df_to_X_preprocessing(df1, cust_ID)
    plt.plot(df1_id[var_x], df1_id[var_y], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red",label=cust_ID)
    plt.legend()

    res = fig
  return res




def process_csv(filename, filter):
    df = pd.read_csv(os.path.join('static/uploadsDB',filename), sep = ',', encoding = 'utf-8')
    df['Power_Level'] = df['Power_Level'].str.strip()
    df['Power_Level'] = df['Power_Level'].str.replace('(\xa0)|(,)', '')
    df['Character'] = df['Character'].str.strip()
    df['Character'] = df['Character'].str.replace('(\xa0)|(,)', '')
    
    df = pd.concat([df, df_add], axis = 0)


    df = df[df['Power_Level'] != '(supressed figting Trunks)']
    df['Power_Level'] = df['Power_Level'].astype(float)
    df = df.sort_values('Power_Level', ascending = False).reset_index(drop = True)
    result = df[(df['Character'] == filter)]
    df_html = result.to_html(classes="table table-striped table-hover",na_rep="-")
    name = result['Character'].iloc[0].replace(',', '')
    img = os.listdir(os.path.join('static', name))[0]
    img_file = os.path.join('static', name, img)
    return df_html, img_file


#SHAP
def sv(data, model, cust_ID):
  explainer = shap.TreeExplainer(model)
  mean_sv = explainer.expected_value[0]
  X = df_to_X_preprocessing(data, cust_ID)
  shap_values = explainer.shap_values(X)[1]
  return shap_values, mean_sv


def global_shap(data, model, nb_features):
  shap_values, mean_sv = sv(data, model, None)
  X = df_to_X_preprocessing(data, None)
  # global shap values for class 1 : bad client
  fig = shap.summary_plot(shap_values, X, max_display=nb_features,show=False)
  return fig


def local_shap(data, model, cust_ID):
  X = df_to_X_preprocessing(data, cust_ID)
  res = X
  if isinstance(res, pd.DataFrame):
    shap_values, mean_sv = sv(data, model, cust_ID)
    # global shap values for class 1 : bad client
    fig = shap.plots._waterfall.waterfall_legacy(mean_sv, # mean of all predictions
                                         shap_values[0,:],
                                         feature_names=res.columns,
                                         show=False)
    res = fig
  return res
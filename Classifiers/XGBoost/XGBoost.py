# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

pd.set_option('display.max_columns', None)


cmap_bold = ListedColormap(['#00FF00', '#FF8000','#FF0000'])

grid_search="no"

escalado="no"

file_name_trainning="DataSet1.csv"
file_name_prediction="Dia17.csv"
ruta=os.getcwd()


df=pd.read_csv(ruta+"/"  + file_name_trainning, sep=';')

print(df.columns)

if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in df.columns:
    df.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)


if 'Axis_Z_positionActualMCS_mm_d10000' in df.columns:
    df.rename(columns={'Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON' }, inplace=True)

if ('N Herramienta' and 'Program Name' and 'Y Motor Power Percent' and 'Override Ejes') in df.columns:
    df=df.drop(['N Herramienta' ,'Y Motor Power Percent','Override Ejes'], axis=1)


df["Posicion Z"]=df["Posicion Z"]/1000
df["Posicion Y"]=df["Posicion Y"]/1000

print("Dataframe de Entrenamiento: \n")
print(df.head())
print("Columnas\n")
print(df.columns)
print("\n")

del(df['Program Name'])
print(df.isna().sum())


if escalado == "si":
    X_scaled=df.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
    

    scaler=MinMaxScaler()
    
    X_scaled[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]=scaler.fit_transform(X_scaled[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']])
    print(X_scaled)

else:    
    X_scaled=df.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]

y=df["Target"]

print("Valores unicos Iniciales\n")
print (pd.unique(df['Target']).tolist())
print("\n")
ro los datos de "train" en entrenamiento y prueba para probar los algoritmos

X_2train, X_new, y_2train, y_new = train_test_split(X_scaled, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_2train, y_2train, test_size=0.2)

#Defino el algoritmo a utilizar
# Create Random Forest Classifier
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=1, max_delta_step=0, max_depth=9,
              min_child_weight=0.5, 
              n_estimators=150, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=0.9,
              tree_method='exact', validate_parameters=1, verbosity=None
)
xgb_clf.fit(X_train, y_train)

score = xgb_clf.score(X_test, y_test)
print(score)

y_pred=xgb_clf.predict(X_test)

from sklearn.metrics import classification_report
print("\nScikitlearn Metrics")
print("Report")
report=classification_report(y_test,y_pred)
print(report)

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import plot_confusion_matrix


print("\nExactitud\n")
exactitud=accuracy_score(y_test, y_pred, normalize=True)
print(exactitud)
print("Precision\n")
precision=precision_score(y_test, y_pred, average='micro')
print(precision)
print("Exhaustividad\n")
exhaust=recall_score(y_test, y_pred, average='micro')
print(exhaust)
print("Valor F\n")
FVal=f1_score(y_test, y_pred, average='micro')
print(FVal)

print("Matriz de Confusion")
matriz=confusion_matrix(y_test, y_pred)
print(matriz)

target_names=["Buen Estado","Mal Estado 1","Mal Estado 2"]

disp=plot_confusion_matrix(xgb_clf, X_test, y_test, display_labels=target_names)

disp.ax_.set_title("XGBoost")

if grid_search == "si":

    param_grid=[{
                
                'learning_rate' : [0.001, 0.01, 0,1],
                'max_depth':[3,5,9],
                'n_estimators': [50, 100, 150],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [0, 0.5, 1],
                #'max_delta_step': [0],
                'subsample': [0.7, 0.8, 0.9, 1],
                #'colsample_bytree': [0.6, 0.8, 1],
                #'reg_alpha': [0, 1e-2, 1, 1e1],
                #'reg_lambda': [0, 1e-2, 1, 1e1],
                #'base_score': [0.5]        
                
               
    }]

    from sklearn.model_selection import GridSearchCV

    grid=GridSearchCV(xgb_clf, param_grid= param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Mejor estimador")
    print(grid.best_estimator_)
    print("Mejor Score")
    print(grid.best_score_)
    print("Mejor parametros")
    print(grid.best_params_)


#####################
# Precision
#####################

fig=plt.figure("Plot")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(df["Posicion Z"], df["Z Motor Power Percent"],df["Presion Contrapeso"], c=df["Target"], cmap=cmap_bold)
#ax.set_title("Datos Entrenamiento prediccion")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contraeso")
ax.set_ylabel("Potencia Z")


fig=plt.figure("Predicciones entrenamiento")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(X_test["Posicion Z"], X_test["Z Motor Power Percent"],X_test["Presion Contrapeso"], c=y_pred, cmap=cmap_bold)

ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contraeso")
ax.set_ylabel("Potencia Z")
plt.show()

###############################
#Prediccion de nuevos dataset
###############################

#Lectura de CSV a predecir
data=pd.read_csv(ruta+"/" + file_name_prediction, sep=',')

#Renombramos las columnas
if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in data.columns:
    data.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)

if 'Axis_Z_positionActualMCS_mm_d10000' in data.columns:
    data.rename(columns={'Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON', 'Axis_FeedRate_actual':'FeedRate Actual' }, inplace=True)

data["Posicion Z"]=data["Posicion Z"]/1000
data["Posicion Y"]=data["Posicion Y"]/1000

if ('N Herramienta' and 'Program Name' and 'Y Motor Power Percent' and 'Override Ejes') in data.columns:
    data=data.drop(['N Herramienta' ,'Y Motor Power Percent','Override Ejes','Program Name','Date', 'Posicion Y', 'FeedRate Actual'], axis=1)


data=data[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso', 'Presion Hidraulica', 'Hidraulics ON']]

print(data.columns)

dataSet= pd.concat([X_new,data])

y2=xgb_clf.predict(dataSet)

fig=plt.figure("Prediccion")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(dataSet["Posicion Z"], dataSet["Z Motor Power Percent"],dataSet["Presion Contrapeso"], c=y2, cmap=cmap_bold)
ax.set_title("XGBoost")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contrapeso")
ax.set_ylabel("Potencia Z")
plt.show()



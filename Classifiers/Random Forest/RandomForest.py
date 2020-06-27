# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # Import Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#00FF00', '#FF8000','#FF0000'])

grid_search= "no"
escalado="no"

file_name_trainning="Pruebas_Trainning_Dataset_3Estados_0SHORT.csv"
#"Dia17_0950a0957.csv"
#'Dia17_22a24.csv'
file_name_prediction=  "Dia17_0950a0957.csv"

path_trainning="G:/OneDrive/V1/data/Trainning/"
path_prediction="G:/OneDrive/V1/data/Prediction/"

df=pd.read_csv(path_trainning + file_name_trainning, sep=';')

print(df.describe())


if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in df.columns:
    df.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)

if 'Axis_Z_positionActualMCS_mm_d10000' in df.columns:
    df.rename(columns={'Cnc_Program_Name_RT':'Program Name','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hydraulics ON' }, inplace=True)


if ('Cnc_Tool_Number_RT' and 'Program Name' and 'Y Motor Power Percent' and 'Cnc_Override_Axis') in df.columns:
    df=df.drop(['Cnc_Tool_Number_RT' ,'Y Motor Power Percent','Cnc_Override_Axis'], axis=1)


print("Dataframe de Entrenamiento: \n")
print(df.head())
print("\n")


X=df.loc[:,['Posicion Z','Z Motor Power Percent','Presion Contrapeso','Presion Hidraulica', 'Hydraulics ON']]
y=df["Target"]

print("Valores unicos Iniciales\n")
print (pd.unique(df['Target']).tolist())
print("\n")


print(df.columns)

print(type(X))

print("Esto es X DataFrame Train:\n")
print(X.head())
print("\n")
print("Esto es Y Dataframe Train:\n")
print(y.head())
print("\n")

from sklearn.preprocessing import MinMaxScaler

#Escalamos los datos de X en DF de trainning
if escalado == "si":

    print(X.columns)
    #Escalado
    #Standarizar los Datos
    scaler = MinMaxScaler()
    X[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hydraulics ON']] =scaler.fit_transform(X[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hydraulics ON']])


else:
    X=X

print("Mostramos el dataframe")
print(X)

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_2train, X_new, y_2train, y_new = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_2train, y_2train, test_size=0.2)

#Defino el algoritmo a utilizar
# Create Random Forest Classifier
regressor = RandomForestClassifier(criterion= 'entropy', max_depth= 7.0, max_features= 'auto', min_samples_split= 0.05)

# Train Decision Tree Classifer
regressor = regressor.fit(X_train,y_train)

#Realizo una prediccion
y_pred = regressor.predict(X_test)

print("Estos son los valores unicos predecidos")
print(np.unique(y_pred))

#####################
# Precision
#####################

from sklearn import metrics

print("#"*50)
print("Teesteo de modelo")
print("#"*50)
print("\n")

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) # Cuanto menor mejor
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Esta es la Precision del Modelo de Testeo de entrenamiento :',accuracy_score(y_test,y_pred.round())*100)
print('\n')



from sklearn.metrics import classification_report
print("Scikitlearn Metrics")
print("\nReport")
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

disp=plot_confusion_matrix(regressor, X_test, y_test, display_labels=target_names)
disp.ax_.set_title("Random Forest")



from sklearn.metrics import confusion_matrix
print("Matriz de Confusion\n")
matriz=confusion_matrix(y_test, y_pred)
print(matriz)

#Hiperparameter Tunning
if grid_search == "si":

    param_grid=[{'criterion':['gini','entropy'],
                'max_depth':[2.0,3.0,4.0,5.0,7.0],
                'max_features' :  ['auto','sqrt','log2'],
                'min_samples_split' : [0.2,0.3,0.1,0.05]
                

    }]
    from sklearn.model_selection import GridSearchCV

    grid=GridSearchCV(regressor, param_grid= param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Mejor estimador")
    print(grid.best_estimator_)
    print("Mejor Score")
    print(grid.best_score_)
    print("Mejor parametros")
    print(grid.best_params_)


#Plot Train DataSet
fig=plt.figure("Plot RAW Trainning Data")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(df["Posicion Z"], df["Z Motor Power Percent"],df["Presion Contrapeso"], c=df["Target"], cmap=cmap_bold)
#ax.set_title("Datos Entrenamiento prediccion")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Counter Weight Pressure")
ax.set_ylabel("Potencia Z")

print(X_test)


fig=plt.figure("Trainning Split")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(X_test["Posicion Z"], X_test["Z Motor Power Percent"],X_test["Presion Contrapeso"],c=y_pred, cmap=cmap_bold)

ax.set_xlabel("Posicion Z")
ax.set_zlabel("Counter Weight Pressure")
ax.set_ylabel("Potencia Z")


#######################################
#Prediccion de nuevos dataset
#########################################

#Lectura de CSV a predecir
data=pd.read_csv(path_prediction + file_name_prediction, sep=',')

#Renombramos las columnas
if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in data.columns:
    data.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)

if 'Axis_Z_positionActualMCS_mm_d10000' in data.columns:
    data.rename(columns={'Cnc_Program_Name_RT':'Program Name','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hydraulics ON'}, inplace=True)

if 'Date' in data.columns:
    del(data["Date"])

print(data.columns)

dataSet= pd.concat([X_new,data])

y2 = regressor.predict(dataSet.loc[:,['Posicion Z','Z Motor Power Percent','Presion Contrapeso','Presion Hidraulica','Hydraulics ON']])

fig=plt.figure("Prediccion")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(dataSet["Posicion Z"], dataSet["Z Motor Power Percent"],dataSet["Presion Contrapeso"], c=y2, cmap=cmap_bold)
ax.set_title("Random Forest")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contrapeso")
ax.set_ylabel("Potencia Z")
plt.show()






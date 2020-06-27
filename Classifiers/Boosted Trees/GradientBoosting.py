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

file_name_trainning="Pruebas_Trainning_Dataset_3Estados_0SHORT.csv"


file_name_prediction="Dia17_0950a0957.csv"

path_trainning="G:/OneDrive/V1/data/Trainning/"
path_prediction="G:/OneDrive/V1/data/Prediction/"

#Lectura de CSV a predecir
data=pd.read_csv(path_prediction + file_name_prediction, sep=',')
#Renombramos las columnas
if 'Axis_Z_positionActualMCS_mm_d10000' in data.columns:
    data.rename(columns={'Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON', 'Axis_FeedRate_actual':'FeedRate Actual' }, inplace=True)


df=pd.read_csv(path_trainning + file_name_trainning, sep=';')

print(df.columns)

if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in df.columns:
    df.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)


if 'Axis_Z_positionActualMCS_mm_d10000' in df.columns:
    df.rename(columns={'Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON' }, inplace=True)

if ('N Herramienta' and 'Program Name' and 'Y Motor Power Percent' and 'Override Ejes') in df.columns:
    df=df.drop(['N Herramienta' ,'Y Motor Power Percent','Override Ejes'], axis=1)


print("Dataframe de Entrenamiento: \n")
print(df.head())
print("Columnas\n")
print(df.columns)
print("\n")


del(df['Program Name'])
print(df.isna().sum())

X=df.loc[:,["Posicion Z", "Z Motor Power Percent", "Presion Contrapeso","Presion Hidraulica", "Hidraulics ON"]]
y=df["Target"]

X_2train, X_new, y_2train, y_new = train_test_split(X, y, test_size=0.2)

if escalado == "si":
    X_scaled=X_2train.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
    data_scaled=data.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]

    scaler=MinMaxScaler()
    
    X_scaled[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]=scaler.fit_transform(X_scaled[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']])
    print(X_scaled)

    data_scaled[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]=scaler.fit_transform(data_scaled[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']])
    print(data_scaled)

else:    
    X_scaled=X_2train.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
    data_scaled=data.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]


y=df["Target"]


print("Valores unicos Iniciales\n")
print (pd.unique(df['Target']).tolist())
print("\n")

print("Esto es X DataFrame Train:\n")
#print(X.head())
print("\n")
print("Esto es Y Dataframe Train:\n")
#print(y.head())
print("\n")

#print(X.columns)

#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_2train, test_size=0.2)

#Defino el algoritmo a utilizar
# Create Random Forest Classifier
gb_clf = GradientBoostingClassifier(learning_rate=0.4 , criterion='mse', max_depth=3, n_estimators=100)
# Train Decision Tree Classifer
gb_clf = gb_clf.fit(X_train,y_train)

#Realizo una prediccion
y_pred = gb_clf.predict(X_test)

#############
# Precisión
#############

#print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


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

disp=plot_confusion_matrix(gb_clf, X_test, y_test, display_labels=target_names)

disp.ax_.set_title("Gradient Boosting")


if grid_search == "si":

    param_grid=[{
                'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5],
                'criterion':['friedman_mse','mse','mae'],
                'max_depth':[3,5,9],
                'n_estimators':[20,40,50,60,100],
                              

    }]

    from sklearn.model_selection import GridSearchCV

    grid=GridSearchCV(gb_clf, param_grid= param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Mejor estimador")
    print(grid.best_estimator_)
    print("Mejor Score")
    print(grid.best_score_)
    print("Mejor parametros")
    print(grid.best_params_)




#####################
# Prediccion
#####################
fig=plt.figure("Plot")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(df["Posicion Z"], df["Z Motor Power Percent"],df["Presion Contrapeso"], c=df["Target"], cmap=cmap_bold)
#ax.set_title("Datos Entrenamiento prediccion")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contrapeso")
ax.set_ylabel("Potencia Z")


fig=plt.figure("Trainning")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(X_test["Posicion Z"], X_test["Z Motor Power Percent"],X_test["Presion Contrapeso"], c=y_pred, cmap=cmap_bold)
#ax.set_title("Datos Entrenamiento prediccion")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion contrapeso")
ax.set_ylabel("Potencia Z")
#plt.show()

###############################
#Prediccion de nuevos dataset
###############################

#Prediccion de datos sin leer
samples=data.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
print(data_scaled["Posicion Z"])

print("Columnas\n")
print(X_new.columns)
print(data_scaled.columns)

print("\n Frames")
print("X_new")
print(X_new.head())
print("Data")
print(data.head())

dataSet= pd.concat([X_new,samples])

y2 = gb_clf.predict(dataSet)

dataSet["Target"]=y2

dataSet.to_csv("Pred_Labeled.csv")


fig=plt.figure("Prediccion Nuevos Datos")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(dataSet["Posicion Z"], dataSet["Z Motor Power Percent"],dataSet["Presion Contrapeso"],c=y2 , cmap=cmap_bold)
ax.set_title("Gradient Boosting")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contrapeso")
ax.set_ylabel("Potencia Z")
plt.show()

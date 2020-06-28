# encoding: utf-8
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#00FF00', '#FF8000','#FF0000'])

grid_search = "no"

#Ruta Actual
ruta=os.getcwd()

file_name_training="DataSet1.csv"
file_name_prediction="Dia17.csv"

#Z son nuestros datos e Y es el Target
df=pd.read_csv(ruta+"/"+ file_name_training, sep=';')

if 'Axis_Z_positionActualMCS_mm_d10000' in df.columns:
                df.rename(columns={'Axis_FeedRate_actual':'FeedRate Actual','Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON' }, inplace=True)


######################################################################################################
#CUIDADO PUEDE NO HACER FALTA HACER LA SIGUIETNE CONVEVERSION DEPENDIENDO LOS DATOS QUE SE USEN
#######################################################################################################
df["Posicion Z"]=df["Posicion Z"]/1000
df["Posicion Y"]=df["Posicion Y"]/1000

######################################################################################################
#
#######################################################################################################

print("Dataframe de Entrenamiento: \n")
print(df.head())
print("\n")

X=X=df.loc[:,['Posicion Z','Posicion Y','Z Motor Power Percent','Presion Contrapeso','FeedRate Actual','Presion Hidraulica','Hidraulics ON' ]]
y=df["Target"]

print("Valores unicos Iniciales\n")
print (pd.unique(df['Target']).tolist())
print("\n")

print("Esto es X DataFrame Train:\n")
print(X.head())
print("\n")
print("Esto es Y Dataframe Train:\n")
print(y.head())
print("\n")



#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_2train, X_new, y_2train, y_new = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_2train, y_2train, test_size=0.2)



#Defino el algoritmo a utilizar
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 2, p= 1, weights= 'uniform')

#Entreno el modelo
knn.fit(X_train, y_train)

#Realizo una predicción
y_pred = knn.predict(X_test)

print("tipo variable salida algoritmo \n")
print(type(y_pred))
#Verifico la matriz de Confusión
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión Modelo Entrenamiento:')
print(matriz)
print("\n")


#Calculo la precisión del modelo
from sklearn.metrics import precision_score
#precision = precision_score(y_test, y_pred)
precision=knn.score(X_test, y_test)
print('Precisión del modelo:')
print(precision)
print("\n")



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

disp=plot_confusion_matrix(knn, X_test, y_test, display_labels=target_names)
disp.ax_.set_title("KNN")


if grid_search == "si":

    param_grid=[{
                'weights': ['uniform', 'distance'],
                'p': [1,2],
                'algorithm':['auto', 'ball_tree', 'kd_tree','brute'],
                'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,14,15,20]
    }]
    from sklearn.model_selection import GridSearchCV

    grid=GridSearchCV(knn, param_grid= param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Mejor estimador")
    print(grid.best_estimator_)
    print("Mejor Score")
    print(grid.best_score_)
    print("Mejor parametros")
    print(grid.best_params_)


fig=plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(df["Posicion Z"], df["Z Motor Power Percent"],df["Presion Contrapeso"], c=y, cmap=cmap_bold)
ax.set_title("Datos Entrenamiento")
ax.set_xlabel("Posicion Z")
ax.set_ylabel("Counter Weight Pressure")
ax.set_zlabel("Potencia Z")
ax.legend()
ax.grid(True)
plt.show()

#Cargamos nuestros datos maqueados pero de un DF que no hayamos utilizado hasta ahora y una simulacion a 25 muestras


#############
# Prediccion
#############


#Lectura de CSV a predecir 
data=pd.read_csv(ruta+"/"+ file_name_prediction, sep=',')
#Renombramos las columnas

if 'Axis_Z_positionActualMCS_mm_d10000' in data.columns:
    data.rename(columns={'Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON', 'Axis_FeedRate_actual':'FeedRate Actual' }, inplace=True)

######################################################################################################
#CUIDADO PUEDE NO HACER FALTA HACER LA SIGUIETNE CONVEVERSION DEPENDIENDO LOS DATOS QUE SE USEN
#######################################################################################################
data["Posicion Z"]=data["Posicion Z"]/1000
data["Posicion Y"]=data["Posicion Y"]/1000

print("DataFrame a predecir")
if "date" in data.columns:
    del(data["Date"])
print(data.head())
print("\n")
print("Los valores nulos del datframe son:\n")
print(df.isnull().sum())

data=data.reset_index()

print(data.columns)
data=data[['Posicion Z','Posicion Y','Z Motor Power Percent','Presion Contrapeso','FeedRate Actual','Presion Hidraulica','Hidraulics ON' ]]

#Escalado
#Standarizar los Datos
#scaler = MinMaxScaler()
#scaled_df =scaler.fit_transform(df_contrapeso_KMeans)
#data = pd.DataFrame(scaler.fit_transform(data), columns=['Posicion Z','Posicion Y','Z Motor Power Percent','Presion Contrapeso'])

#Sin Escalado
#print(np.isfinite(data).all())
#print(np.argwhere(np.isnan(data)))

dataSet= pd.concat([X_new,data])

y2_pred=knn.predict(dataSet)

dataSet["Target"]=y2_pred

print("\nDataFrame Con la predicion echa:\n")
print(dataSet.head())
print("Tail")
print(dataSet.tail())
print("\n")
print("Valores unicos finales\n")
print (pd.unique(dataSet['Target']).tolist())
print("\n")

fig=plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(dataSet["Posicion Z"], dataSet["Z Motor Power Percent"],dataSet["Presion Contrapeso"], c=y2_pred, cmap=cmap_bold)
ax.set_title("KNN")
ax.set_xlabel("Posicion Z")
ax.set_zlabel("Presion Contrapeso")
ax.set_ylabel("Potencia Z")
plt.show()









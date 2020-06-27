# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cmap_bold = ListedColormap(['#00FF00', '#FF8000','#FF0000'])

#Z son nuestros datos e Y es el Target


escalado="no"
grid_search="no"


file_name_trainning="Pruebas_Trainning_Dataset_3Estados_0SHORT.csv"
file_name_prediction="Dia17_0950a0957.csv"

path_trainning="G:/OneDrive/V1/data/Trainning/"
path_prediction="G:/OneDrive/V1/data/Prediction/"



df=pd.read_csv(path_trainning + file_name_trainning, sep=';')

print(df.columns)

if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in df.columns:
    df.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)

if 'Axis_Z_positionActualMCS_mm_d10000' in df.columns:
    df.rename(columns={'Cnc_Program_Name_RT':'Program Name','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso', 'System_IOLINK_HydraulicPressure':'Presion Hidraulica', 'System_isHydraulicsOn':'Hidraulico ON'}, inplace=True)



print("Dataframe de Entrenamiento: \n")
print(df.head())
print("\n")
print(df.columns)

df['Target'].astype(np.int64)

X=df.loc[:,["Posicion Z", "Z Motor Power Percent", "Presion Contrapeso","Presion Hidraulica", "Hidraulico ON"]]
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


if escalado == "si":

    #Escalado
    #Standarizar los Datos
    scaler = MinMaxScaler()
    X_scaled =scaler.fit_transform(X)
    #data = pd.DataFrame(scaler.fit_transform(data), columns=['Posicion Z','Posicion Y','Z Motor Power Percent','Presion Contrapeso'])
else:

    X_scaled=X


#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_2train, X_new, y_2train, y_new = train_test_split(X_scaled, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_2train, y_2train, test_size=0.2)

#Tengo que meter el GridSearchAntes


#Defino el algoritmo a utilizar
# Create Decision Tree classifer object

clf = DecisionTreeClassifier( criterion='entropy', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=5, random_state=None, max_leaf_nodes=6)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Realizo una prediccion
y_pred = clf.predict(X_test)

#Evaluación precisión
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) # Cuanto menor mejor
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Esta es la Precision del Modelo de Testeo de entrenamiento :',accuracy_score(y_test,y_pred.round())*100)
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

disp=plot_confusion_matrix(clf, X_test, y_test, display_labels=target_names)
disp.ax_.set_title("Decision Tree")


if grid_search == "si":

    param_grid=[{'criterion':['gini','entropy'],
                'max_depth':[2,3,4,5,6,7,8],
                'max_features' :  [1,2,3,4,5,6,7],
                'max_leaf_nodes' : [1,2,3,4,5,6]
                

    }]
    from sklearn.model_selection import GridSearchCV

    grid=GridSearchCV(clf, param_grid= param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Mejor estimador")
    print(grid.best_estimator_)
    print("Mejor Score")
    print(grid.best_score_)
    print("Mejor parametros")
    print(grid.best_params_)


fig=plt.figure("Trainning Dataset Plot")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(df["Posicion Z"], df["Z Motor Power Percent"],df["Presion Contrapeso"], c=df["Target"], cmap=cmap_bold)
#ax.set_title()
ax.set_xlabel("Posicion Z")
ax.set_ylabel("Potencia Z")
ax.set_zlabel("Presion Contrapeso")

if escalado == "no":
    fig=plt.figure("Prediccion Datos de entrenamiento")
    ax= fig.add_subplot(111, projection='3d')
    ax.scatter(X_test["Posicion Z"], X_test["Z Motor Power Percent"],X_test["Presion Contrapeso"], c=y_pred, cmap=cmap_bold)
    #ax.set_title()
    ax.set_xlabel("Posicion Z")
    ax.set_ylabel("Potencia Z")
    ax.set_zlabel("Presion Contrapeso")
    #plt.show()


###############################
#Prediccion de nuevos dataset
###############################

#Lectura de CSV a predecir
data=pd.read_csv(path_prediction + file_name_prediction, sep=',')

#Renombramos las columnas

if 'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE' in data.columns:
    data.rename(columns={'PLCPostPgm|-|AXES[2].DG_ACTUAL_VALUE':'Posicion Z','PLCPostPgm|-|AXES[1].DG_ACTUAL_VALUE':'Posicion Y','PLCPostPgm|-|DG_V_MOTOR_POWER':'Z Motor Power Percent', 'PLCPostPgm|-|WG_COUNTERWEIGHTPRESSURE':'Presion Contrapeso'}, inplace=True)


if 'Axis_Z_positionActualMCS_mm_d10000' in data.columns:
    data.rename(columns={'Cnc_Program_Name_RT':'Program Name','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Z Motor Power Percent','Axis_Z_power_percent':'Y Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso', 'System_IOLINK_HydraulicPressure':'Presion Hidraulica', 'System_isHydraulicsOn':'Hidraulico ON'}, inplace=True)



if 'Date' in data.columns:
    del(data["Date"])

if 'Program Name' in data.columns:
    del(data['Program Name'])

if 'Y Motor Power Percent' in data.columns:
    del(data['Y Motor Power Percent'])    

if 'Posicion Y' in data.columns:
    del(data['Posicion Y'])    


if 'Cnc_Tool_Number_RT' in data.columns:
    del(data['Cnc_Tool_Number_RT'])    
if 'Cnc_Override_Axis' in data.columns:    
    del(data['Cnc_Override_Axis'])   
if 'Axis_FeedRate_actual' in data.columns:
    del(data['Axis_FeedRate_actual'])   



if 'Time' in data.columns:
    del(data['Time'])  


print(data.columns)

if escalado == "si":

    #Escalado
    #Standarizar los Datos
   
    data_scaled =scaler.fit_transform(data[['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso', 'Presion Hidraulica', 'Hidraulico ON']])
    #data = pd.DataFrame(scaler.fit_transform(data), columns=['Posicion Z','Posicion Y','Z Motor Power Percent','Presion Contrapeso'])

else:
    data_scaled=data


dataSet= pd.concat([X_new,data_scaled])

y2 = clf.predict(dataSet)

fig=plt.figure("Prediccion Nuevo Dataset")
ax= fig.add_subplot(111, projection='3d')
ax.scatter(dataSet["Posicion Z"], dataSet["Z Motor Power Percent"],dataSet["Presion Contrapeso"], c=y2, cmap=cmap_bold)
ax.set_title("Decision Tree")
ax.set_xlabel("Posicion Z")
ax.set_ylabel("Potencia Z")
ax.set_zlabel("Presion Contrapeso")
plt.show()


#################################################
#Para visualizacion de Arbol y obtener una imagen
#################################################

# Visualize Decision Tree
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
import pydotplus

# Creates dot file named tree.dot
#dot_data=StringIO()
dot_data=tree.export_graphviz(
            clf,
            out_file =  None,
            feature_names = list(X.columns),
            class_names = ['0', '1'],
            filled = True,
            rounded = True)


#graph=graphviz.Source(dot_data)
graph=pydotplus.graph_from_dot_data(dot_data)

#graph.render('dtree_render', view=True)

Image(graph.create_png())

# Create PDF
graph.write_pdf("Images/" + file_name_trainning + ".pdf")

# Create PNG
graph.write_png("Images/" + file_name_trainning + "_PNG.png")





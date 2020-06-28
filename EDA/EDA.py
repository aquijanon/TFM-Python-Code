# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import os

#Definimos los colores
cmap_bold = ListedColormap([ '#00FF00', '#FF8000','#FF0000' ])

#Para Mostrar todas las columnas en el Output de la Terminal
pd.set_option('display.max_columns', None)

#Ruta Actual
ruta=os.getcwd()
#Se declaran variables
file_path='DataSet1.csv'   
df_contrapeso=pd.read_csv(ruta+"/" + file_path, sep=';')
#Para activar desactivar pandas profiling (0/1)
pd_profiling=0 

#Exploratory Data Analysis [EDA]
print( "#"*50)
print("Exploratory Data Analysis")
print( "#"*50+ "\n")    

#Renombramos Columnas
if 'Axis_Z_positionActualMCS_mm_d10000' in df_contrapeso.columns:
        df_contrapeso.rename(columns={',Date':'Date','Axis_FeedRate_actual':'Avance','Cnc_Program_Name_RT':'Program Name', 'Cnc_Tool_Number_RT': 'N Herramienta', 'Cnc_Override_Axis':'Override Ejes','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Carga Motor Z','Axis_Z_power_percent':'Carga Motor Y', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON' }, inplace=True)

#Convertimos los valores de las posiciones a mm dividiendo /1000
df_contrapeso["Posicion Z"]=df_contrapeso["Posicion Z"]/10000
df_contrapeso["Posicion Y"]=df_contrapeso["Posicion Y"]/10000

if 'Time' in df_contrapeso.columns:
        df_contrapeso_corr=df_contrapeso.drop(['Time'], axis=1)
else:
        df_contrapeso_corr=df_contrapeso
                      

fig=plt.figure("SCATTER PLOT")
ax= fig.add_subplot(111, projection='3d')
p=ax.scatter(df_contrapeso["Posicion Z"],df_contrapeso["Presion Contrapeso"],df_contrapeso["Carga Motor Z"], c= df_contrapeso["Target"], cmap=cmap_bold)
ax.set_xlabel("Posicion Z")
ax.set_ylabel("Presión Contrapeso")
ax.set_zlabel("Potencia Z")
fig.colorbar(p)
plt.show()


#Creamos el Reporte
if pd_profiling == 1:
        print("Pandas profiling\n")

        report=df_contrapeso.profile_report()
        report.to_file("report_"+file_path[:-4]+".html")

print("\nLos tipos de variable que hay en el DataFrame son: ")
print(df_contrapeso.dtypes)
print("\n")

print('Dataframe que hemos cargado')
print (df_contrapeso.head())
print("\n")

#Obtención de los nulos en todo el dataframe.
print("El numero de nulos en el DataFrame es de: ",df_contrapeso.isna().sum())
print("\n") 

print("Descripcion del data frame")
dsc=df_contrapeso.describe()
print(type(dsc))
print(dsc)

df_contrapeso.drop(columns=df_contrapeso.columns[df_contrapeso.nunique()==1], inplace=True)

print("Descripción del DataFrame: ")
print(df_contrapeso_corr.describe())
print("\n")

df_contrapeso_corr=df_contrapeso_corr.drop(["Override Ejes", "Target", "Posicion Y", "Date","Program Name"], axis=1)

#Sacamos el nombre de las Columnas que tiene el df
col_names=list(df_contrapeso_corr.columns)
print('Nombre Columnas DF')
print(col_names)
print("\n") 

#Hacemos la matriz de correlación entre las variables.
correlacion= df_contrapeso_corr.corr(method='pearson')
print("La correlación entre las variables es de: ")
print(correlacion)
print("\n")

#HEATMAP PLOT
fig=plt.figure("HEATMAP")
correlacion.style.background_gradient(cmap='coolwarm').set_precision(2)
print("Las variables en el heatmap son: ",col_names)

ax=plt.gca()
im=ax.matshow(correlacion)
fig.colorbar(im)     
ax.set_xticks(np.arange(len(col_names))) #Sin esto no muestra el primer label!!!
ax.set_xticklabels(col_names, rotation=45)
ax.set_yticks(np.arange(len(col_names)))
ax.set_yticklabels(col_names)
#Para mostrar el valor de la correlación en el HeatMap
for (i, j), z in np.ndenumerate(correlacion):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
      
print(df_contrapeso.columns)

#Identificación de outliers Visualmente
Data=[df_contrapeso["Posicion Z"],
        df_contrapeso["Presion Contrapeso"],
        df_contrapeso["Carga Motor Z"],
        df_contrapeso["Presion Hidraulica"],df_contrapeso["Avance"]]
        
fig,a=plt.subplots(2,3,squeeze=False, num= 'BOXPLOT')

a[0][0].boxplot(Data[0])
a[0][0].set_title('Posicion Z')
a[0][1].boxplot(Data[1])
a[0][1].set_title('Presion Contrapeso')
a[0][2].boxplot(Data[2])
a[0][2].set_title('Z Motor Power')      
a[1][0].boxplot(Data[3])
a[1][0].set_title('Presión Hidraulica')
a[1][2].boxplot(Data[4])
a[1][2].set_title('Avance')
  
#Distribución de los datos
bin_length=5

fig,a=plt.subplots(2,3,squeeze=False, num='HISTOGRAMA' )

bin_PosZ=int((Data[0].max()-Data[0].min())/bin_length)
a[0][0].hist(Data[0], color = 'blue', edgecolor = 'black',
        bins = 10)
a[0][0].set_title('Posicion Z')

bin_Presion=int((Data[1].max()-Data[1].min())/bin_length)
a[0][1].hist(Data[1], color = 'blue', edgecolor = 'black',
        bins = 20)
a[0][1].set_title('Presion Contrapeso')

bin_ZPower=int((Data[2].max()-Data[2].min())/bin_length)
a[0][2].hist(Data[2], color = 'blue', edgecolor = 'black',
        bins =10)
a[0][2].set_title('Z Motor Power')

bin_GH=int((Data[3].max()-Data[3].min())/bin_length)
a[1][0].hist(Data[3], color = 'blue', edgecolor = 'black',
        bins = 10)
a[1][0].set_title('Presion hidraulica')

bin_GHON=int((Data[4].max()-Data[4].min())/bin_length)
a[1][1].hist(Data[4], color = 'blue', edgecolor = 'black',
        bins = 4)
a[1][1].set_title('Avance')

plt.show()





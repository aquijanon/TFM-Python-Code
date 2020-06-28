# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_profiling
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import dash_dangerously_set_inner_html
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import base64

print('\n')
print('Bienvenido de nuevo :)')
print('\n')

global filex

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]


############
## Dash
############
def visualizacion():
    print('###########################')
    print('Seccion de visualizacion')
    print('###########################')
    print('\n')

    app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)
    app.config["suppress_callback_exceptions"] = True
    
    
    #Rutas de interes
    image_filename = 'files/images/Soraluce_Logo.png' # replace with your own image
    carpeta="data/"
    dicc={}

    #Encoding Logo Soraluce
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())

    #Buscamos en la carpeta todos los dataset que hay
    for archivo in os.listdir(carpeta):
        dicc.update({archivo : archivo})
       
        if os.path.isdir(os.path.join(carpeta,archivo)):
            os.path.join(carpeta,archivo)
        
    #Creamos el Banner
    def build_banner():
        return html.Div(
            id="banner",
            className="banner",
            children=[
                html.Div(
                    id="banner-text",
                    children=[
                        html.H1("Mantenimiento Predictivo"),
                        html.H2("Monitorización de Señales y algoritmos de Machine Learning"),
                    ],
                ),
                html.Div(
                    id="banner-logo",
                    children=[
                        
                        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
                     
                    ],
                ),
                dcc.Dropdown(
                            id="csv-list", options=[{'label': i, 'value': i} for i in dicc],value=dicc["Dia17 _0857a0906.csv"]
                        ),
                html.Div(id="OpcionesConfig"),
                html.Div(id="hidden-div", style={'display': 'none'}),
                html.Div(id="hidden-div1", style={'display': 'none'})

            ],
        )


    def profiling(df_contrapeso,filex):

        print("profiling")


        if os.path.isfile(filex) == False : #Sino existe el archivo en la ruta
            report=df_contrapeso.profile_report()
            report.to_file(filex+".html")
            report_html=report.to_html()
            
        else:

            print("Hola Mundo")
   

    app.layout = html.Div(children=[
    build_banner(),
    dcc.Tabs(id="tabs-Principal", value='tab-1-V', children=[
        dcc.Tab(label='Visualizacion', value='tab-1-V'),
        dcc.Tab(label='Machine Learning', value='tab-2-ML')
    ]),

    html.Div(id='pestanas'),#Aqui es donde se mostrarían todos los datos
    #dcc.Graph(id='grafico1')
    
    ]   
    )


    @app.callback(
    Output('pestanas','children'),
    [Input('tabs-Principal','value'),Input('csv-list','value')])
    def pestania(tab,filex):

        df_contrapeso= lectura (filex)
        print(df_contrapeso.head())
        print("\n")
        correlacion= eda(df_contrapeso)
        cortolist=correlacion.values.tolist()
        print("Correlacion: \n")
        print(correlacion)
        print("\n")
        print(type(correlacion))
        print("\n")
        

        if tab == 'tab-1-V':#Seleccion pestaá de visualizacion
            print("Hola 1")


            return html.Div(id="visualizacion", children=[

                html.H1("Obtener reporte de Pandas Profiling"),
                
                html.Button('Obtener', id='btn1'),
                #html.Div(dash_table.DataTable(id = 'table', data=[dict(desc)])),
                html.Div(id='PandasProfiling'),
                

                
                html.H1("EDA"),
                html.H2("Histograma"),
                dcc.Graph(id='graph-1-tabs',
                    figure={
                        'data': [{
                            #'x': [1,1,1, 2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,5,5,6],
                            'x': df_contrapeso['Presion Contrapeso'],
                            'type': 'histogram',
                            'nbinsx' : 20,
                            'name': 'Presion Contrapeso'
                        },
                        ],
                        'layout':{  
                            'title' :'Presion Contrapeso' 
                        }        
                    },    
                    ),

                dcc.Graph(id='graph-2-tabs',
                figure={
                    'data': [{
                        #'x': [1,1,1, 2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,5,5,6],
                        'x': df_contrapeso['Posicion Y'],
                        'type': 'histogram',
                        'nbinsx' : 20,
                        'name': 'Posicion Y'
                    },
                    ],
                    'layout':{ 
                        
                        'title' :'Posicion Y' 
                    }        
                },      
                ),

                dcc.Graph(id='graph-3-tabs',
                figure={
                    'data': [{
                        #'x': [1,1,1, 2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,5,5,6],
                        'x': df_contrapeso['Posicion Z'],
                        'type': 'histogram',
                        'nbinsx' : 15,
                        'name': 'Posicion Z'
                    },
                    
                    ],
                    'layout':{ 
                        
                        'title' :'Posicion Z'    
                    }          
                },      
                ),

                #Heat Map
                html.H2("HeatMap"),

                dcc.Graph(
                    id='heatmap',
                        figure={
                            'data': [{
                                'x': ["Posicion Z", "Posicion Y", "Z Motor Power Percent", "CounterWeight Pressure"],

                                'y': ["Posicion Z", "Posicion Y", "Z Motor Power Percent", "CounterWeight Pressure"],

                                'z': cortolist,
                            
                                'type': 'heatmap',
                            
                            
                            }]
                        }
                 ),


               html.H2("BoxPlots"),
               dcc.Graph(
                   id='BoxPlot1',
                    figure={
                        'data': [{
                            'x': df_contrapeso["Presion Contrapeso"],
                            'type': 'box',   
                        }],
                        'layout':{ 
                        'title' :'Presion Contrapeso'
                        }
                        
                    }
                ),
                dcc.Graph(
                   id='BoxPlot2',
                    figure={
                        'data': [{
                            'x': df_contrapeso["Posicion Z"],
                            'type': 'box',   
                        }],
                        'layout':{ 
                        'title' :'Posicion Z'
                        }
                        
                    }
                ),
                dcc.Graph(
                   id='BoxPlot3',
                    figure={
                        'data': [{
                            'x': df_contrapeso["Posicion Y"],
                            'type': 'box',   
                        }],
                        'layout':{ 
                        'title' :'Posicion Y'
                        }  
                    }
                )

            ])

        elif tab == 'tab-2-ML':
            print("Hola2")

            return html.Div(id='machine learning', children=[
                
                html.Div(id="Conjunto_Entrenamiento"),
                html.Div(id="Conjunto_Prediccion"),
                html.Div(id="hidden-div2", style={'display': 'none'}),
                machine_learning(df_contrapeso),
                
            ])

    @app.callback(
    Output('PandasProfiling','children'),
    [Input('btn1','n_clicks'),Input('csv-list','value')])
    def Reporting_dash(n_clicks,filex):

        df_contrapeso= lectura (filex)

      
        print( " Dash Reporting")


        if n_clicks == None:
            print("No se ha pulsado el boton")
            
        if n_clicks == 1:
            print("Se ha pulsado el boton")
            profiling(df_contrapeso, filex)
            print("Finish")
            return html.Div([html.H1('Reporte Creado')])
        

    @app.callback(
    Output('Conjunto_Entrenamiento','children'),
    [Input('tabs-Principal','value'),Input('csv-list','value')])
    def Machine_learning(tab,filex):

        df_contrapeso=lectura(filex)
        machine_learning(df_contrapeso)

        # Llamamos a una funcion de ML
        if tab== "tab-2-ML":
            return html.Div([html.H1("Machine Learning")])

    @app.callback(
    Output('hidden-div2','children'),
    [Input('btnReporte','n_clicks'),Input('csv-list','value')])
    def Reporte(n_clicks, pathFile):
        
        if n_clicks == None:
            print("No se ha pulsado el boton")
            #return html.Div([html.H1('Veamos')])
        if n_clicks == 1:
            print("Se ha pulsado el boton")
            ReportGenerator()
            print("Finish")
            return html.Div([html.H1('Reporte Creado')])

 
    if __name__ == '__main__' :
        app.run_server(debug=True)

    return pathFile


###########################
#Lectura 
###########################
def lectura (pathFile):
    print('#####################################################')
    print('#######Entrada en la seccion de Lectura ##############')
    print('#####################################################')
    print("\n")
    df_contrapeso=pd.read_csv("data/"+pathFile, sep=',')

    
    #Renombramos Columnas
   
    if 'Axis_Z_positionActualMCS_mm_d10000' in df_contrapeso.columns:
        df_contrapeso.rename(columns={'Cnc_Program_Name_RT':'Program Name','Axis_Z_positionActualMCS_mm_d10000':'Posicion Z','Axis_Y_positionActualMCS_mm_d10000':'Posicion Y','Axis_Y_power_percent':'Y Motor Power Percent','Axis_Z_power_percent':'Z Motor Power Percent', 'System_IOLINK_CounterweightPressure':'Presion Contrapeso','System_IOLINK_HydraulicPressure': 'Presion Hidraulica','System_isHydraulicsOn':'Hidraulics ON' }, inplace=True)

    #Convertimos los valores de las posiciones a mm dividiendo /1000 a mm
    df_contrapeso["Posicion Z"]=df_contrapeso["Posicion Z"]/1000
    df_contrapeso["Posicion Y"]=df_contrapeso["Posicion Y"]/1000
    return df_contrapeso


###########################
# EDA
###########################
def eda(df_contrapeso):

    #Exploratory Data Analysis [EDA]

    #Para Mostrar todas las columnas en el Output de la Terminal
    pd.set_option('display.max_columns', None)

    print("los tipos de variable que hay en el DataFrame son: ")
    print(df_contrapeso.dtypes)
    print("\n")

    print('df_contrapeso cabecera')
    print (df_contrapeso.head())
    print("\n")

    #Obtención de los nulos en todo el dataframe.
    print("El numero de nulos en el DataFrame es de: ",df_contrapeso.isna().sum())
    print("\n") 

    #Para Saber que numeros aparecen en el sensor de presion
    #unicos= list(sorted(df_contrapeso['Presion Contrapeso'].unique()))

    #Nos deshacemos de variables que no necesitamos y creamos un nuevo dataFrame.
    #df_contrapeso_corr=df_contrapeso.drop(['Time','Target'], axis=1)

    #Para Dataset original
    if 'Time' in df_contrapeso.columns:
        df_contrapeso_corr=df_contrapeso.drop(['Time'], axis=1)
    else:
        df_contrapeso_corr=df_contrapeso

    #Sacamos el nombre de las Columnas que tiene el df
    col_names=list(df_contrapeso_corr.columns)
    print('Nombre Columnas DF')
    print(col_names)
    print("\n") 

    print("Descripción: ")
    print(df_contrapeso_corr.describe())
    print("\n")

    #Hacemos la matriz de correlación entre las variables.
    correlacion= df_contrapeso_corr.corr(method='pearson')
    print("La correlación entre las variables es de: ")
    print(correlacion)
    print("\n")


    #HEATMAP PLOT
    fig=plt.figure()
    
    correlacion.style.background_gradient(cmap='coolwarm').set_precision(2)
    print("Las columnas son: ",col_names)
    print(col_names[:])

    ax=plt.gca()
    im=ax.matshow(correlacion)
    fig.colorbar(im)
    
    ax.set_xticks(np.arange(len(col_names))) #Sin esto no muestra el primer label!!!
    ax.set_xticklabels(col_names, rotation=45)
    ax.set_yticks(np.arange(len(col_names)))
    ax.set_yticklabels(col_names)

    for (i, j), z in np.ndenumerate(correlacion):#Para mostrar el valor de la correlación en el HeatMap
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    #plt.show()

    


    #Identificación de outliers Visualmente
    Data=[df_contrapeso_corr["Posicion Z"],
                df_contrapeso_corr["Presion Contrapeso"],
                df_contrapeso_corr["Z Motor Power Percent"]]

    
    fig,a=plt.subplots(1,3,squeeze=False)
    a[0][0].boxplot(Data[0])
    a[0][0].set_title('Posicion Z')
    a[0][1].boxplot(Data[1])
    a[0][1].set_title('Presion Contrapeso')
    a[0][2].boxplot(Data[2])
    a[0][2].set_title('Z Motor Power')

    #plt.show()

    
    #Distribución de los datos
    bin_length=5

    fig,a=plt.subplots(1,3,squeeze=False)

    bin_PosZ=int((Data[0].max()-Data[0].min())/bin_length)
    a[0][0].hist(Data[0], color = 'blue', edgecolor = 'black',
            bins = 5)
    a[0][0].set_title('Posicion Z')

    bin_Presion=int((Data[1].max()-Data[1].min())/bin_length)
    a[0][1].hist(Data[1], color = 'blue', edgecolor = 'black',
            bins = 3)
    a[0][1].set_title('Presion Contrapeso')

    bin_ZPower=int((Data[2].max()-Data[2].min())/bin_length)
    a[0][2].hist(Data[2], color = 'blue', edgecolor = 'black',
            bins =4)
    a[0][2].set_title('Z Motor Power')

    #plt.show()

    #KDE
    print("KDE \n")

    #kde=df_contrapeso_corr.plot.kde()
    return correlacion
    

###########################
# Machine Learning
###########################
def machine_learning(df_contrapeso):
    from sklearn.ensemble import GradientBoostingClassifier
    print("Machine Learning")

    df_contrapeso_trainning= pd.read_csv("data/DataSet_1.csv", sep=';')

    #X=df_contrapeso.loc[:,['Posicion Z', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
    X=df_contrapeso_trainning[['Posicion Z','Posicion Y', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
    print(X.columns)
    y=df_contrapeso_trainning["Target"]

    #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf = GradientBoostingClassifier(learning_rate=1 , criterion='friedman_mse', max_depth=5, n_estimators=20, max_features=2)
    # Train Decision Tree Classifer
    gb_clf = gb_clf.fit(X_train,y_train)

    #Realizo una prediccion
    y_pred = gb_clf.predict(X_test)

    #Prediccion
    X_new=df_contrapeso[['Posicion Z','Posicion Y', 'Z Motor Power Percent', 'Presion Contrapeso','Presion Hidraulica', 'Hidraulics ON']]
    
    new_pred=gb_clf.predict(X_new)

    pred_df=pd.DataFrame(new_pred, columns=['Target'])

    pred_df.to_csv("files/predicciones/predicciones.csv")



    return html.Div([html.H1("Generar Reporte"),
                html.Button('Reporte', id='btnReporte'),
            
        
                dcc.Graph(
                    id='ML-ScatterPlot1',
                            figure={
                                'data': [{
                                'x': X_train["Posicion Z"],#PosZ 
                                'y': X_train["Presion Contrapeso"],#Presion Contrapeso
                                'z': X_train["Z Motor Power Percent"],#Consumo motor
                                'type': 'scatter3d',
                                'mode':'markers', 
                                'marker':{ 'size' : 3, 'color':y_train, 'showscale': True,}, 
                            }],
                                'layout':{ 
                                'title' :'Datos Entrenamiento',
                                'xaxis':{'title':{'text': 'my title'}}
                            },
                            
                    }

                ),
                dcc.Graph(
                    id='ML-ScatterPlot2',
                            figure={
                                'data': [{
                                'x': X_test["Posicion Z"],#PosZ 
                                'y': X_test["Presion Contrapeso"],#Presion Contrapeso
                                'z': X_test["Z Motor Power Percent"],#Consumo motor
                                'type': 'scatter3d',
                                'mode':'markers', 
                                'marker':{ 'size' : 3, 'color':y_pred, 'showscale': True,}, 
                            }],
                                'layout':{ 
                                'title' :'Datos Test Predicciones',
                                'xaxis':{'title':{'text': 'my title'}}
                            },
                            
                    }

                ),

                dcc.Graph(
                    id='ML-ScatterPlot3',
                            figure={
                                'data': [{
                                'x': X_new["Posicion Z"],#PosZ 
                                'y': X_new["Presion Contrapeso"],#Presion Contrapeso
                                'z': X_new["Z Motor Power Percent"],#Consumo motor
                                'type': 'scatter3d',
                                'mode':'markers', 
                                'marker':{ 'size' : 3, 'color':new_pred, 'showscale': True,}, 
                            }],
                                'layout':{ 
                                'title' :'Nuevo Conjunto de datos',
                                'xaxis':{'title':{'text': 'my title'}}
                            },
                            
                    }

                ),
                
                



            ])


###########################
# Generar Reporte
###########################
def ReportGenerator ():
    print("Report Generator")

    from reportlab.pdfgen import canvas
    from reportlab.lib import colors, utils
    from reportlab.graphics import renderPDF
    from datetime import date
    from io import BytesIO
    from svglib.svglib import svg2rlg
    import matplotlib.pyplot as plt
    from matplotlib import pylab, mlab
    from datetime import date, timedelta
    import numpy as np

    from reportlab.lib.colors import HexColor


    ############################################
    #Fechas 
    ############################################
    hoy=date.today()

    print("Fecha de hoy\n")
    print(hoy)

    print("Fecha de hoy\n")
    print(hoy)
    #Establecemos rango de fechas ficticio
    day1=date(2008,8,15)
    day2=date(2009,8,14)

    fecha1=date(2020,6,17)

    delta=day2-day1
    fechas=[]

    ##################
    #Contenido Report
    ##################
    #Declaración de variables de caracteristicas de pagina
    tam_pag=(1060,1447)#Unidad Puntos (Igual que la de los informes de lakber)
    tope=1350
    entrelineas=20
    margen_izq=50
    margen_izq2=margen_izq+entrelineas
    margen_izq3=margen_izq2+entrelineas
    margen_estados=350
    fuente ='Times-Roman'
    fuente_bold='Times-Bold'
    size_texto=16
    size_titulos=26
    #Declariacion de variables de texto
    file_name= "informes/Informe.pdf"
    tituloDocumento= "Reporte contrapeso"
    titulo="Diagnóstico de máquina. Condición del contrapeso"
    subtitulo="Análisis Contrapeso"
    imagen="files/images/LogoDash.png"
    grafico="files/Report Graphs/Report_Graph.png"
    grafico_resized = "files/Report Graphs/Report_Graph_resized.png"

    ref_maquina='TA-M 3046'
    cliente='Stock Soraluce'

    historicos=pd.read_csv('files/historicos/EstadoComponenete Historicos.csv', sep=';')



    print(historicos)


    texto =['Fecha del informe: '+str(fecha1), 
    'Podrían existir datos que aún no se han transmitido a la plataforma. Podría afectar a la información que se muestra.']

    texto2 =[ref_maquina + '                                                   ' + cliente]


    #Leemos los datos para hacer la condicion
    data=pd.read_csv("files/predicciones/predicciones.csv", sep=',')
    tam=len(data)
    print(tam)

    valor_umbral=15

    umbral= (tam*valor_umbral)/100



    ceros=(data["Target"] == 0).sum()
    malo1=(data["Target"] == 1).sum()
    malo2=(data["Target"] == 2).sum()



    if (malo1+malo2)>umbral:

        print("estado malo")

        condicion=2

    elif (malo1+malo2)<umbral and (malo1+malo2)>0:
        print("Estado satisfactorio")
        condicion=1
    elif (malo1+malo2) == 0:
        print("Buena condicion")
        condicion=0


    #condicion=1#Variable que debe de ser borrada en un futura ya que la lo leera de la salida del algoritmo de machine learning

    print("la condicion actual es")
    print(condicion)

    historicos['Target Contrapeso']=historicos['Target Contrapeso'].replace(0, 0.5)
    historicos['Target Contrapeso']=historicos['Target Contrapeso'].replace(1, 1.5)
    historicos['Target Contrapeso']=historicos['Target Contrapeso'].replace(2, 2.5)


    #Creamos ya los objetos pdf en python
    pdf= canvas.Canvas(file_name, pagesize=tam_pag)
    pdf.setTitle(tituloDocumento)

    #Creamos imagen Sorlauce Arriba a la izquierda
    pdf.drawInlineImage(imagen, margen_izq, tope)

    #Seteamo el tamaño y fuente del titulo
    pdf.setFont(fuente_bold, size_titulos, leading = None) 
    #Titulo le pasamos las coordenadas x,y 0,0 es abajo a la izquierda y el texto del titulo usando drawCentredString centraremos el titulo
    pdf.drawString(margen_izq,tope- entrelineas, titulo)

    text=pdf.beginText(margen_izq,tope-2*entrelineas)
    text.setFont(fuente,size_texto)
    text.setFillColor(colors.black)

    for line in texto:
        text.textLine(line)

    pdf.drawText(text)

    pdf.setFont(fuente_bold, size_titulos, leading = None) 
    pdf.drawString(margen_izq2,tope-(4*entrelineas+entrelineas),"Identificación de Máquina")

    text2=pdf.beginText(margen_izq2,tope-(6*entrelineas+entrelineas))
    text2.setFont(fuente,size_texto)

    for line2 in texto2:
        text2.textLine(line2)

    pdf.drawText(text2)

    pdf.drawText(text)
    pdf.setFont(fuente_bold, size_titulos, leading = None) 
    pdf.drawString(margen_izq2,tope-(8*entrelineas+entrelineas),"Análisis Basado en Condición")

    pdf.drawString(margen_izq2,tope-(10*entrelineas+entrelineas),"Condición del Contrapeso")

    pdf.drawText(text)
    pdf.setFont(fuente, size_texto, leading = None) 
    pdf.drawString(margen_izq3,tope-13*entrelineas,"Contrapeso")

    #PAra mostrar el estado actual de la maquina
    if condicion == 0:
        pdf.setFillColorRGB(0,255,0)
        pdf.setFont(fuente_bold, size_texto, leading = None) 
        pdf.drawString(margen_estados,tope-13*entrelineas,"BUENO")

    elif condicion == 1:
        pdf.setFillColorRGB(255, 139, 0)#(255,165,0) 
        pdf.setFont(fuente_bold, size_texto, leading = None) 
        pdf.drawString(margen_estados,tope-13*entrelineas,"SATISFACTORIO")

    elif condicion == 2:
        pdf.setFillColorRGB(255,0,0)
        pdf.setFont(fuente_bold, size_texto, leading = None) 
        pdf.drawString(margen_estados,tope-13*entrelineas,"MALO")


    ###########################
    #Construccion del gráfico
    ###########################

    #Insertamos el grafico de matplotlib
    hoy=date.today()

    for i in range(delta.days + 1):
        day = day1 + timedelta(days=i)
        #print(day)
        if day.day == 15:
            
            fechas.append(day.isoformat())

    print(fechas)
    print(len(fechas))
    print(type(fechas))

    #Definimos los datos a mostrar
    lista = [0,0.3,0.9,1.6,2.8,0,0.5,0.8,0.8,1,1.6,1.8]
    intervalo_anio = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

    #Creamos el Grafico
    fig=plt.figure("Gráfica Report")
    plt.plot(historicos["Target Contrapeso"], marker='o')
    plt.axhspan(0, 1, alpha=0.3, color='#70de5d', label="Buena")
    plt.axhspan(1, 2, alpha=0.3, color='#ff8b38', label="Satisfactoria")#FFA500
    plt.axhspan(2, 3, alpha=0.3, color='#ff4242', label="Mala")
    plt.title("Condición Contrapeso")
    plt.xticks(intervalo_anio, historicos["Fecha"], rotation=30, ha="right")
    plt.legend()
    plt.subplots_adjust(bottom=0.15)#Aumenta el margen inferior para que entre toda la fecha
    plt.savefig('files/Report Graphs/Report_Graph.png')

    #Con la figura de matplotlib creada se guarda en byutes para despues introducirla directamente al report
    imgdata = BytesIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data

    drawing=svg2rlg(imgdata)

    renderPDF.draw(drawing,pdf, 30, 570)

    #Guardamos el documento pdf creado
    pdf.save()
    print("\nPDF guardado")



visualizacion()




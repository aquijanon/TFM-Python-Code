# -*- coding: utf-8 -*-
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
import pandas as pd
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
file_name= "Reporte.pdf"
tituloDocumento= "Reporte contrapeso"
titulo="Diagnóstico de máquina. Condición del contrapeso"
subtitulo="Análisis Contrapeso"
imagen="Logo.png"
grafico="Report Graphs/Report_Graph.png"
grafico_resized = "Report Graphs/Report_Graph_resized.png"

ref_maquina='TA-M 3046'
cliente='Stock Soraluce'

historicos=pd.read_csv('EstadoComponenete Historicos.csv', sep=';')

print(historicos)

texto =['Fecha del informe: '+str(fecha1), 
'Podrían existir datos que aún no se han transmitido a la plataforma. Podría afectar a la información que se muestra.']

texto2 =[ref_maquina + '                                                   ' + cliente]

#Leemos los datos para hacer la condicion
data=pd.read_csv("Pred_Labeled.csv", sep=',')
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
plt.savefig('Report Graphs/Report_Graph.png')


#Con la figura de matplotlib creada se guarda en byutes para despues introducirla directamente al report
imgdata = BytesIO()
fig.savefig(imgdata, format='svg')
imgdata.seek(0)  # rewind the data

drawing=svg2rlg(imgdata)

renderPDF.draw(drawing,pdf, 30, 600)

#Guardamos el documento pdf creado
pdf.save()
print("\nPDF guardado")
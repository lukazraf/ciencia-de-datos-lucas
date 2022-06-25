import matplotlib.pyplot  as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats 
import warnings
import pandas as pd

df=pd.read_csv('lucas20.csv')
pd.read_csv('lucas20.csv')
df.head(30)
df.tail(30)
df[['id','price']]
st.write('**Datos de king country, USA (20xx a 20xx)**:')
df=pd.read_csv('C:/Users\KAROL/Desktop/app lucas ciencia de datos/Lucas20.csv')
st.dataframe(df)

st.write('**Casas en el periodo analisado:**','Disponibles {} casas'.format(data['id'].nunique()))

st.write('**Casa mas barata:**')
st.dataframe(df[df['price']==df['price'].min()])

st.write('**Casa mas cara:**')
st.dataframe(df[df['price']==df['price'].max()])

st.title('Filtros')
##forma de seleccionarlos los datos para el filtro
OptFiltro = st.multiselect(
     'Que quieres filtrar',
     ['Precios','Habitaciones', 'Baños', 'Metros cuadrados (Espacio habitable)','Pisos','Vista al mar','Indice de construccion','Condicion'],
     ['Habitaciones', 'Baños','Pisos'])


from cmath import nan
import pandas as pd
import seaborn as sns
import numpy as np
from plotly import express as px
from matplotlib import pyplot as plt 
from matplotlib import gridspec
from sqlalchemy import false, true
import streamlit as st 
st.write('**Casa mas barata:**')
st.dataframe(df)
st.write('**Casas en el periodo analisado:**','Disponibles {} casas'.format(df['id'].nunique()))

df=pd.read_csv('lucas20.csv')
pd.read_csv('lucas20.csv')
df.head(30)
df.tail(30)
df[['id','price']]

df[['id','price', 'view']]
df['price'].mean
df['price'].describe
df.shape
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats 
import warnings


sns.distplot(df['price'])
df['price'].skew
df['price'].kurt
var='sqft_living'
df=pd.concat([df['price'], df['sqft_living']])
df.head(20)
df.plot.scatter(x=var, y='price', ylim=(0,10000000))
var='sqft_basement'
df=pd.concat([df['price'], df['sqft_basement']])
df.plot.scatter(x=var, y='price', ylim=(0,10000000))
f, ax=plt.subplots(figsize=(40,6))
fig=sns.boxplot(x='yr_built', y='price', df= df)
fig.axis(ymin=0, ymax=800000)
var='yr_renovated'
df.head
df=pd.concat([df['price'], df['yr_renovated']])
df.plot.scatter(x=var, y='price', ylim=(0,10000000))
f, ax=plt.subplots(figsize=(40,6))
fig=sns.boxplot(x='yr_renovated', y='price', df= df)
fig.axis(ymin=0, ymax=5000000)
st.title('Aplicacion de prueba')

st.write('**Datos de king country, USA (20xx a 20xx)**:')
df=pd.read_csv('C:/Users\KAROL/Desktop/app lucas ciencia de datos/Lucas20.csv')
st.dataframe(df)

st.write('**Casas en el periodo analisado:**','Disponibles {} casas'.format(data['id'].nunique()))

st.write('**Casa mas barata:**')
st.dataframe(df[df['price']==df['price'].min()])

st.write('**Casa mas cara:**')
st.dataframe(df[df['price']==df['price'].max()])

st.title('Filtros')
##forma de seleccionarlos los datos para el filtro
OptFiltro = st.multiselect(
     'Que quieres filtrar',
     ['Precios','Habitaciones', 'Baños', 'Metros cuadrados (Espacio habitable)','Pisos','Vista al mar','Indice de construccion','Condicion'],
     ['Habitaciones', 'Baños','Pisos'])


import pandas as pd

from cmath import nan
import pandas as pd
import seaborn as sns
import numpy as np
from plotly import express as px
from matplotlib import pyplot as plt 
from matplotlib import gridspec
from sqlalchemy import false, true
import streamlit as st 

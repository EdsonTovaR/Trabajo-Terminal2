import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz


#Conjunto de datos
def Conjunto_Datos():
    data = pd.read_csv("tested.csv")
    return data

#Exploracion de los datos
def Exploracion_Datos(data):
    print("Exploración de los datos")
    print(data.head())
    print(data.describe())
    print(data.info())
    
#Identificacion de variables irrelevantes
def Identificacion(data):
    #Borramos las tablas que no sirvan como Nombres
    data = data.drop(['Nombre', 'Boleto', 'Sexo', 'Cabina', 'Embarcado'], axis=1)
    
    #usamos el promedio en los valores que tiene NaN
    data = data.fillna(data.mean())
  
    
    arbol = DecisionTreeRegressor()
    arbol.fit(data.iloc[:, :-1], data.iloc[:, -1])
    
    #Importancia de las variables
    importacias = arbol.feature_importances_
    print(importacias)
    return data

#Evaluamos la calidad de los datos
def Calidad_Datos(data):
    #Verificamos si hay datos nulos
    print(data.isnull().sum())
    #Verificamos si hay datos duplicados
    print(data.duplicated().sum())
    #Verificamos si hay datos atipicos
    print(data.describe())
    #Verificamos si hay datos unicos
    print(data.nunique())
    #Verificamos si hay datos en blanco
    print(data.isnull().sum())
    
    return 

#Guardamos los datos limpios y los mostramos con graphviz
def Guardar_Datos(data):
    data.to_csv("cleaned_data.csv", index=False)
    return



    
if __name__ == "__main__":
    print("CONJUNTO DE DATOS: ")
    data = Conjunto_Datos()
    print("EXPLORACION DE DATOS")
    Exploracion_Datos(data)
    print("IDENTIFICACION DE DATOS")
    Identificacion(data)
    Guardar_Datos(data)
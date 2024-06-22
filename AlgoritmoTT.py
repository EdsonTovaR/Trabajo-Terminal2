import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaner")
        self.root.geometry("600x500")
        self.root.configure(bg='black')
        self.data = None

        # Interfaz gráfica
        self.label = tk.Label(root, text="Seleccionar archivo", width=40, height=10, bg="blue", fg="white", font=("century gothic", 14))
        self.label.pack(padx=20, pady=20)
        self.label.bind("<Button-1>", self.load_file)

        self.clean_button = tk.Button(root, text="Limpiar Datos", command=self.clean_data, bg="navy", fg="white", font=("century gothic", 14))
        self.clean_button.pack(padx=20, pady=20)
        self.clean_button.config(state=tk.DISABLED)

    def load_file(self, event):
        file_path = filedialog.askopenfilename()
        
        
        if file_path:
            self.data = pd.read_csv(file_path)
            self.label.config(text="Archivo cargado: " + file_path.split("/")[-1])
            self.clean_button.config(state=tk.NORMAL)
            
            
       
    def clean_data(self):
        #borramos las tablas que no sirvan y no sean un float o number
        self.data.dropna(inplace=True)
        self.data = self.data.select_dtypes(include=['float64', 'int64'])
        self.data.to_csv("cleaned_data.csv", index=False)
            
        #guardamos el archivo
        #self.data.to_csv("cleaned_data.csv", index=False)
        
        if self.data is not None:
            self.impute_missing_values()
            self.label.config(text="Datos limpiados y guardados como 'cleaned_data.csv'")
            self.data.to_csv("cleaned_data.csv", index=False)
            

            
            

    def impute_missing_values(self):
        # Imputar valores faltantes usando árbol de decisión
        for column in self.data.columns:
            if self.data[column].isnull().any():
                self.impute_column(column)

    def impute_column(self, column):
        # Crear un conjunto de datos para entrenar el árbol de decisión
        df = self.data.copy()
        train = df[df[column].notnull()]
        test = df[df[column].isnull()]

        if not test.empty:
            # Seleccionar características y objetivo
            X_train = train.drop(columns=[column])
            y_train = train[column]
            X_test = test.drop(columns=[column])

            # Entrenar el árbol de decisión
            tree = DecisionTreeRegressor()
            tree.fit(X_train, y_train)

            # Predecir los valores faltantes
            predicted_values = tree.predict(X_test)
            self.data.loc[self.data[column].isnull(), column] = predicted_values

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()

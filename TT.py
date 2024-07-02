import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from tkinter import filedialog, ttk, scrolledtext

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaner")
        self.root.geometry("800x400")
        self.root.configure(bg='black')
        self.data = None

        # Interfaz gráfica
        
        #Boton para seleccionar archivos del almacenamiento
        self.label = tk.Label(root, text="Seleccionar archivo", width=40, height=10, bg="blue", fg="white", font=("century gothic", 14))
        self.label.pack(padx=20, pady=20)
        self.label.bind("<Button-1>", self.cargar_datos)
        
        #Boton para limpiar datos
        self.clean_button = tk.Button(root, text="Limpiar Datos", command=self.limpiar_datos, bg="navy", fg="white", font=("century gothic", 14))
        self.clean_button.pack(padx=20, pady=20)
        self.clean_button.config(state=tk.DISABLED)

        #Boton para guardar datos
        self.save_button = tk.Button(root, text="Guardar Datos", command=self.guardar_datos, bg="green", fg="white", font=("century gothic", 14))
        self.save_button.pack(padx=20, pady=20)
        self.save_button.config(state=tk.DISABLED)
        
        #Barra de progreso
        self.progress = ttk.Progressbar(root, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=50)
        
        #Boton para seleccionar las columnas
        self.column_button = tk.Button(root, text="Seleccionar Columnas", command=self.seleccion_columnas, bg="blue", fg="white", font=("century gothic", 14))
        self.column_button.pack(padx=20, pady=20)
        
        #Boton para seleccionar las opciones de limpieza    
        self.cleaning_button = tk.Button(root, text="Opciones de Limpieza", command=self.opciones_limpieza, bg="blue", fg="white", font=("century gothic", 14))
        self.cleaning_button.pack(padx=20, pady=20)
        
        #Boton para mostrar datos que solo se mostrara al ejecutar la funcion seleccionar columnas
        self.summary_button = tk.Button(root, text="Resumen de Datos", command=self.mostrar_resumen, bg="blue", fg="white", font=("century gothic", 14))
        self.summary_button.pack(padx=20, pady=20)
        self.summary_button.config(state="disabled")
        
         # Cuadro de texto para mostrar los datos de las columnas seleccionadas
        self.data_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("century gothic", 12))
        self.data_text.pack(padx=20, pady=20)
        self.data_text.config(state=tk.DISABLED)
        
        # Inicialmente oculta la Listbox
        self.column_listbox = None

        
        
    def cargar_datos(self, event):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.data = pd.read_csv(file_path)
                self.label.config(text="Archivo cargado: " + file_path.split("/")[-1])
                self.clean_button.config(state=tk.NORMAL)
        except Exception as e:
            self.label.config(text=f"Error al cargar el archivo: {str(e)}")
            self.clean_button.config(state=tk.DISABLED)
            
    def guardar_datos(self):
        try:
            self.data.to_csv("cleaned_data.csv", index=False)
            self.label.config(text="Datos guardados como 'cleaned_data.csv'")
        except Exception as e:
            self.label.config(text=f"Error al guardar el archivo: {str(e)}")
            
    #funcion para que el usuario seleccione las columnas a revisar
    def seleccion_columnas(self):
        self.column_listbox = tk.Listbox(self.root, bg="red", selectmode=tk.MULTIPLE)
        self.column_listbox.pack(pady=10)
        for column in self.data.columns:
            self.column_listbox.insert(tk.END, column)
        self.column_button.config(state="disabled")
        self.summary_button.config(state="normal")
        #Mostramos las columnas de la base de datos y lo mandamos a la funcion mostrar_resumen
        self.mostrar_resumen()
    
        
        
    def mostrar_resumen(self):
        selected_indices = self.column_listbox.curselection()
        self.selected_columns = [self.data.columns[i] for i in selected_indices]
        
        # Mostrar datos de las columnas seleccionadas
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        
        if self.selected_columns:
            selected_data = self.data[self.selected_columns]
            self.data_text.insert(tk.END, selected_data.to_string(index=False))
        else:
            self.data_text.insert(tk.END, "No hay columnas seleccionadas.")
        
        self.data_text.config(state=tk.DISABLED)
        
        
    #opciones de limpieza con tecnicas de imputacion y preprocesamiento
    def opciones_limpieza(self):
        self.strategy_var = tk.StringVar(value="mean")
        self.strategy_menu = tk.OptionMenu(self.root, self.strategy_var, "Promedio", "Mediana", "Más_Frecuente", "KNN", "Iterative")
        self.strategy_menu.pack(pady=10)
        self.cleaning_button.config(state="disabled")
        
    def limpiar_datos(self):
        
        return

    

        
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()
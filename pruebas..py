
import pandas as pd
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
import numpy as np

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaner")
        self.root.geometry("600x600")
        self.root.configure(bg='black')
        self.data = None

        # Interfaz gráfica
        self.label = tk.Label(root, text="Seleccionar archivo", width=40, height=10, bg="blue", fg="white", font=("century gothic", 14))
        self.label.pack(padx=20, pady=20)
        self.label.bind("<Button-1>", self.load_file)

        self.clean_button = tk.Button(root, text="Limpiar Datos", command=self.clean_data, bg="navy", fg="white", font=("century gothic", 14))
        self.clean_button.pack(padx=20, pady=20)
        self.clean_button.config(state=tk.DISABLED)
        
        self.save_button = tk.Button(root, text="Guardar Datos", command=self.save_data, bg="green", fg="white", font=("century gothic", 14))
        self.save_button.pack(padx=20, pady=20)
        self.save_button.config(state=tk.DISABLED)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=400, mode='indeterminate')
        self.progress.pack(pady=10)

    def load_file(self, event):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.data = pd.read_csv(file_path)
                self.label.config(text="Archivo cargado: " + file_path.split("/")[-1])
                self.clean_button.config(state=tk.NORMAL)
        except Exception as e:
            self.label.config(text=f"Error al cargar el archivo: {str(e)}")
            self.clean_button.config(state=tk.DISABLED)
            
            return file_path
                    
            
            
            
            
            

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()


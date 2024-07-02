import tkinter as tk
from tkinter import filedialog, ttk, simpledialog
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaner")
        self.root.geometry("1200x800")
        self.root.configure(bg='sky blue')
        self.data = None

        # Interfaz gráfica
        self.label = tk.Label(root, text="Seleccionar archivo", width=40, height=2, bg="blue", fg="white", font=("century gothic", 14))
        self.label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)
        self.label.bind("<Button-1>", self.cargar_datos)
        
        # Frame para los botones
        button_frame = tk.Frame(root, bg='black')
        button_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=20)
        
        self.clean_button = tk.Button(button_frame, text="Limpiar Datos", command=self.limpiar_datos, bg="navy", fg="white", font=("century gothic", 14))
        self.clean_button.grid(row=0, column=0, padx=10, pady=10)
        self.clean_button.config(state=tk.DISABLED)

        self.save_button = tk.Button(button_frame, text="Guardar Datos", command=self.guardar_datos, bg="green", fg="white", font=("century gothic", 14))
        self.save_button.grid(row=0, column=1, padx=10, pady=10)
        self.save_button.config(state=tk.DISABLED)
        
        self.show_button = tk.Button(button_frame, text="Mostrar Datos", command=self.mostrar_datos, bg="blue", fg="white", font=("century gothic", 14))
        self.show_button.grid(row=0, column=2, padx=10, pady=10)
        self.show_button.config(state=tk.DISABLED)
        
        self.cleaning_button = tk.Button(button_frame, text="Opciones de Limpieza", command=self.mostrar_menu_limpieza, bg="blue", fg="white", font=("century gothic", 14))
        self.cleaning_button.grid(row=1, column=0, padx=10, pady=10)
        self.cleaning_button.config(state=tk.DISABLED)
        
        self.suggest_button = tk.Button(button_frame, text="Sugerir Limpieza", command=self.sugerir_limpieza, bg="orange", fg="white", font=("century gothic", 14))
        self.suggest_button.grid(row=1, column=1, padx=10, pady=10)
        self.suggest_button.config(state=tk.DISABLED)
        
        self.anomaly_button = tk.Button(button_frame, text="Detectar Anomalías", command=self.detectar_anomalias, bg="red", fg="white", font=("century gothic", 14))
        self.anomaly_button.grid(row=1, column=2, padx=10, pady=10)
        self.anomaly_button.config(state=tk.DISABLED)
        
        self.train_button = tk.Button(button_frame, text="Entrenar Modelo", command=self.entrenar_modelo, bg="blue", fg="white", font=("century gothic", 14))
        self.train_button.grid(row=2, column=0, padx=10, pady=10)
        self.train_button.config(state=tk.DISABLED)
        
        self.graph_button = tk.Button(button_frame, text="Mostrar Gráficos", command=self.mostrar_graficos, bg="blue", fg="white", font=("century gothic", 14))
        self.graph_button.grid(row=2, column=1, padx=10, pady=10)
        self.graph_button.config(state=tk.DISABLED)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=800, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=2, padx=20, pady=20)
        
        self.data_text = tk.Text(root, width=120, height=10, bg="white", fg="black", font=("century gothic", 12))
        self.data_text.grid(row=3, column=0, columnspan=2, padx=20, pady=20)
        self.data_text.config(state=tk.DISABLED)

    def cargar_datos(self, event):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.data = pd.read_csv(file_path)
                self.label.config(text="Archivo cargado: " + file_path.split("/")[-1])
                self.clean_button.config(state=tk.NORMAL)
                self.show_button.config(state=tk.NORMAL)
                self.cleaning_button.config(state=tk.NORMAL)
                self.suggest_button.config(state=tk.NORMAL)
                self.anomaly_button.config(state=tk.NORMAL)
                self.train_button.config(state=tk.NORMAL)
                self.graph_button.config(state=tk.NORMAL)
        except Exception as e:
            self.label.config(text=f"Error al cargar el archivo: {str(e)}")
            self.clean_button.config(state=tk.DISABLED)
            
    def guardar_datos(self):
        try:
            self.data.to_csv("cleaned_data.csv", index=False)
            self.label.config(text="Datos guardados como 'cleaned_data.csv'")
        except Exception as e:
            self.label.config(text=f"Error al guardar el archivo: {str(e)}")
            
    def mostrar_datos(self):
        if self.data is not None:
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, f"Resumen de los datos:\n\n")
            self.data_text.insert(tk.END, f"Columnas: {self.data.shape[1]}\n")
            self.data_text.insert(tk.END, f"Filas: {self.data.shape[0]}\n\n")
            self.data_text.insert(tk.END, f"{self.data.head()}")
            self.data_text.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)
        
    def mostrar_menu_limpieza(self):
        def limpiar_opcion(opcion):
            if opcion == "Eliminar Nulos":
                self.data = self.data.dropna()
            elif opcion == "Rellenar Nulos con Media":
                self.data = self.data.fillna(self.data.mean())
            self.mostrar_datos()
        
        opciones = ["Eliminar Nulos", "Rellenar Nulos con Media"]
        seleccion = simpledialog.askstring("Opciones de Limpieza", "Selecciona una opción:", initialvalue=opciones[0])
        limpiar_opcion(seleccion)

    def sugerir_limpieza(self):
        if self.data is not None:
            missing_values = self.data.isnull().sum().sum()
            unique_values = self.data.nunique().sum()
            
            sugerencias = []
            
            if missing_values > 0:
                sugerencias.append(f"Hay {missing_values} valores faltantes.")
            
            if unique_values / self.data.size < 0.05:
                sugerencias.append("Algunas columnas pueden tener baja diversidad en los datos.")
            
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, "\n".join(sugerencias) + "\n")
            self.data_text.config(state=tk.DISABLED)
    
    def detectar_anomalias(self):
        if self.data is not None:
            clf = IsolationForest(contamination=0.1)
            self.data['anomaly'] = clf.fit_predict(self.data.select_dtypes(include=[float, int]))
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, f"Anomalías detectadas:\n{self.data[self.data['anomaly'] == -1]}\n")
            self.data_text.config(state=tk.DISABLED)
            
    def entrenar_modelo(self):
        if self.data is not None:
            X = self.data.select_dtypes(include=[float, int]).drop(columns=['anomaly'], errors='ignore')
            y = self.data.iloc[:, -1]
            model = LinearRegression()
            model.fit(X, y)
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, f"Modelo entrenado con coeficientes: {model.coef_}\n")
            self.data_text.config(state=tk.DISABLED)

    def mostrar_graficos(self):
        if self.data is not None:
            self.data.hist(figsize=(10, 6))
            plt.show()
            
    def limpiar_datos(self):
        pass
    
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()

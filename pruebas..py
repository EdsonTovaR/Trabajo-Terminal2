import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaner")
        self.root.geometry("1200x800")
        self.root.configure(bg='sky blue')
        self.data = None
        self.data_cleaned = None

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
        
        self.train_button = tk.Button(button_frame, text="Entrenar Modelo", command=self.entrenar_modelo, bg="blue", fg="white", font=("century gothic", 14))
        self.train_button.grid(row=2, column=0, padx=10, pady=10)
        self.train_button.config(state=tk.DISABLED)
        
        self.graph_button = tk.Button(button_frame, text="Mostrar Gráficos", command=self.mostrar_graficos, bg="blue", fg="white", font=("century gothic", 14))
        self.graph_button.grid(row=2, column=1, padx=10, pady=10)
        self.graph_button.config(state=tk.DISABLED)

        self.tree_button = tk.Button(button_frame, text="Crear Árbol", command=self.seleccionar_columnas_arbol, bg="blue", fg="white", font=("century gothic", 14))
        self.tree_button.grid(row=2, column=2, padx=10, pady=10)
        self.tree_button.config(state=tk.DISABLED)
        
        self.tree_button = tk.Button(button_frame, text="Ver confiabilidad de datos", command=self.evaluar_confiabilidad, bg="blue", fg="white", font=("century gothic", 14))
        self.tree_button.grid(row=2, column=2, padx=10, pady=10)
        self.tree_button.config(state=tk.DISABLED)

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
                self.train_button.config(state=tk.NORMAL)
                self.graph_button.config(state=tk.NORMAL)
                self.tree_button.config(state=tk.NORMAL)
                #self.evaluar_confiabilidad(self.data, "Antes de la limpieza")
        except Exception as e:
            self.label.config(text=f"Error al cargar el archivo: {str(e)}")
            self.clean_button.config(state=tk.DISABLED)
            
    def guardar_datos(self):
        try:
            self.data_cleaned.to_csv("cleaned_data.csv", index=False)
            self.label.config(text="Datos guardados como 'cleaned_data.csv'")
        except Exception as e:
            self.label.config(text=f"Error al guardar el archivo: {str(e)}")
            
    def mostrar_datos(self):
        if self.data_cleaned is not None:
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, f"Resumen de los datos:\n\n")
            self.data_text.insert(tk.END, f"Columnas: {self.data_cleaned.shape[1]}\n")
            self.data_text.insert(tk.END, f"Filas: {self.data_cleaned.shape[0]}\n\n")
            self.data_text.insert(tk.END, f"{self.data_cleaned.head()}")
            self.data_text.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)
    
    def mostrar_menu_limpieza(self):
        def limpiar_opcion(opcion, columna):
            if opcion == "Eliminar Nulos":
                self.data_cleaned[columna] = self.data_cleaned[columna].dropna()
            elif opcion == "Rellenar Nulos con Media":
                media = self.data_cleaned[columna].mean()
                self.data_cleaned[columna] = self.data_cleaned[columna].fillna(media)

        # Crear la ventana principal oculta
        root = tk.Tk()
        root.withdraw()

        # Copiar datos para limpiar
        self.data_cleaned = self.data.copy()

        # Iterar sobre cada columna
        for columna in self.data.columns:
            seleccion = messagebox.askquestion(
                f"Opciones de Limpieza para '{columna}'",
                f"Selecciona una opción para la columna '{columna}':",
                type=messagebox.YESNO,
                default=messagebox.YES,
                icon=messagebox.QUESTION,
                detail="Sí: Eliminar Nulos\nNo: Rellenar Nulos con Media"
            )
            
            if seleccion == 'yes':
                limpiar_opcion("Eliminar Nulos", columna)
            elif seleccion == 'no':
                limpiar_opcion("Rellenar Nulos con Media", columna)
        
        self.evaluar_confiabilidad(self.data_cleaned, "Después de la limpieza")
        self.mostrar_datos()
        self.graph_button.config(state=tk.NORMAL)
            
    def sugerir_limpieza(self):
        if self.data is not None:
            missing_values = self.data.isnull().sum().sum()
            unique_values = self.data.nunique().sum()
            
            sugerencias = []
            
            if missing_values > 0:
                sugerencias.append(f"Hay {missing_values} valores faltantes.")
            else:
                sugerencias.append("No hay valores faltantes.")
            
            if unique_values / self.data.size < 0.05:
                sugerencias.append("Algunas columnas pueden tener baja diversidad en los datos.")          
            
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, "\n".join(sugerencias) + "\n")
            self.data_text.config(state=tk.DISABLED)
    
    def entrenar_modelo(self):
        if self.data_cleaned is not None:
            X = self.data_cleaned.select_dtypes(include=[float, int]).drop(columns=['anomaly'], errors='ignore')
            y = self.data_cleaned.iloc[:, -1]
            model = DecisionTreeClassifier()
            model.fit(X, y)
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete("1.0", tk.END)
            self.data_text.insert(tk.END, f"Modelo entrenado con importancia de características: {model.feature_importances_}\n")
            self.data_text.config(state=tk.DISABLED)

    def mostrar_graficos(self):
        if self.data_cleaned is not None:
            self.data_cleaned.hist(figsize=(10, 6))
            plt.show()
            
    def seleccionar_columnas_arbol(self):
        column_window = tk.Toplevel(self.root)
        column_window.title("Seleccionar Columnas")
        column_window.geometry("400x400")

        label = tk.Label(column_window, text="Selecciona columnas para el Árbol de Decisión:", font=("century gothic", 14))
        label.pack(pady=10)

        columns = self.data.columns
        self.column_vars = {col: tk.BooleanVar() for col in columns}

        for col in columns:
            chk = tk.Checkbutton(column_window, text=col, variable=self.column_vars[col])
            chk.pack(anchor=tk.W)

        button_frame = tk.Frame(column_window)
        button_frame.pack(pady=20)
        
        select_all_button = tk.Button(button_frame, text="Seleccionar Todo", command=self.seleccionar_todo)
        select_all_button.grid(row=0, column=0, padx=5)

        deselect_all_button = tk.Button(button_frame, text="Deseleccionar Todo", command=self.deseleccionar_todo)
        deselect_all_button.grid(row=0, column=1, padx=5)

        create_button = tk.Button(column_window, text="Crear Árbol", command=self.crear_arboles)
        create_button.pack(pady=10)

    def seleccionar_todo(self):
        for var in self.column_vars.values():
            var.set(True)

    def deseleccionar_todo(self):
        for var in self.column_vars.values():
            var.set(False)

    def procesar_datos(self, X):
        # Codificar variables categóricas
        X_encoded = pd.get_dummies(X, drop_first=True)
        return X_encoded

    def crear_arboles(self):
        if self.data_cleaned is not None:
            selected_columns = [col for col, var in self.column_vars.items() if var.get()]
            if selected_columns:
                X = self.data_cleaned[selected_columns]
                y = self.data_cleaned.iloc[:, -1]  # Suponiendo que la última columna es la variable objetivo
                X_encoded = self.procesar_datos(X)  # Codificar variables categóricas
                model = DecisionTreeClassifier()
                model.fit(X_encoded, y)
                self.data_text.config(state=tk.NORMAL)
                self.data_text.delete("1.0", tk.END)
                self.data_text.insert(tk.END, f"Árbol de decisión creado con importancia de características: {model.feature_importances_}\n")
                self.data_text.config(state=tk.DISABLED)
            else:
                messagebox.showwarning("Advertencia", "Selecciona al menos una columna para crear el árbol.")

    def evaluar_confiabilidad(self, data, estado):
        # Recuento de valores faltantes
        missing_values = data.isnull().sum().sum()

        # Conteo de valores únicos
        unique_counts = data.nunique()

        # Estadísticas descriptivas
        descriptive_stats = data.describe(include='all')

        # Mostrar información
        self.data_text.config(state=tk.NORMAL)
        self.data_text.insert(tk.END, f"\nConfiabilidad de datos - {estado}:\n")
        self.data_text.insert(tk.END, f"Valores faltantes: {missing_values}\n")
        self.data_text.insert(tk.END, "Valores únicos por columna:\n")
        self.data_text.insert(tk.END, f"{unique_counts}\n")
        self.data_text.insert(tk.END, f"Estadísticas descriptivas:\n{descriptive_stats}\n")
        self.data_text.config(state=tk.DISABLED)

    def limpiar_datos(self):
        return
            
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()

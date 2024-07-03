import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Cleaner")
        self.root.geometry("1200x800")
        self.root.configure(bg='sky blue')
        self.data = None
        self.selected_columns = []

        # Interfaz gráfica
        self.label = tk.Label(root, text="Seleccionar archivo", width=40, height=2, bg="blue", fg="white", font=("century gothic", 14))
        self.label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)
        self.label.bind("<Button-1>", self.cargar_datos)
        
        # Frame para los botones
        button_frame = tk.Frame(root, bg='black')
        button_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=20)
        

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
        
        self.graph_button = tk.Button(button_frame, text="Mostrar Gráficos", command=self.mostrar_graficos, bg="blue", fg="white", font=("century gothic", 14))
        self.graph_button.grid(row=2, column=1, padx=10, pady=10)
        self.graph_button.config(state=tk.DISABLED)

        self.tree_button = tk.Button(button_frame, text="Crear Árbol", command=self.seleccionar_columnas_arbol, bg="blue", fg="white", font=("century gothic", 14))
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
                
                self.show_button.config(state=tk.NORMAL)
                self.cleaning_button.config(state=tk.NORMAL)
                self.suggest_button.config(state=tk.NORMAL)
                self.graph_button.config(state=tk.NORMAL)
                self.tree_button.config(state=tk.NORMAL)
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
            self.data_text.insert(tk.END, f"{self.data}")
            self.data_text.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)
        
    def mostrar_menu_limpieza(self):
        def limpiar_opcion(opcion, columna):
            if opcion == "Eliminar Nulos":
                self.data[columna] = self.data[columna].dropna()
            elif opcion == "Rellenar Nulos con Media":
                media = self.data[columna].mean()
                self.data[columna] = self.data[columna].fillna(media)

        # Crear la ventana principal oculta
        root = tk.Tk()
        root.withdraw()

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

    def mostrar_graficos(self):
        if self.data is not None:
            self.data.hist(figsize=(10, 6))
            plt.show()
            
    
    def seleccionar_todo(self):
        for var in self.column_vars.values():
            var.set(True)

    def deseleccionar_todo(self):
        for var in self.column_vars.values():
            var.set(False)

    def guardar_seleccion(self):
        self.selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        messagebox.showinfo("Selección de Columnas", f"Columnas seleccionadas: {self.selected_columns}")
        #Buscamos que numero de columna son las columnas elegidas
        columnas = [self.data.columns.get_loc(col) for col in self.selected_columns]
        X = self.data.iloc[:, columnas]
        print(X)
        y = self.data.iloc[:, -1]
        print(y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        plt.figure(figsize=(20, 10))
        plot_tree(clf, filled=True, feature_names=self.selected_columns)
        plt.show()
        

    def seleccionar_columnas_arbol(self):
        if self.data is not None:
            self.column_vars = {col: tk.BooleanVar() for col in self.data.columns}
            
            # Crear la ventana principal oculta
            root = tk.Tk()
            root.withdraw()
            
            # Crear la ventana de selección de columnas
            columnas_window = tk.Toplevel()
            columnas_window.title("Seleccionar Columnas para el Árbol")
            columnas_window.geometry("800x600")
            
            # Crear el frame de las columnas
            columnas_frame = tk.Frame(columnas_window)
            columnas_frame.pack(padx=20, pady=20)
            
            # Crear los botones
            select_all_button = tk.Button(columnas_frame, text="Seleccionar Todo", command=self.seleccionar_todo)
            select_all_button.grid(row=0, column=0, padx=10, pady=10)
            
            deselect_all_button = tk.Button(columnas_frame, text="Deseleccionar Todo", command=self.deseleccionar_todo)
            deselect_all_button.grid(row=0, column=1, padx=10, pady=10)
            
            save_button = tk.Button(columnas_frame, text="Guardar Selección", command=self.guardar_seleccion)
            save_button.grid(row=0, column=2, padx=10, pady=10)
            
            # Crear los checkboxes
            for i, col in enumerate(self.data.columns):
                cb = tk.Checkbutton(columnas_frame, text=col, variable=self.column_vars[col])
                cb.grid(row=i+1, column=0, columnspan=3, padx=10, pady=10)
                
            columnas_window.mainloop()
            
            
            

    
    
    
    
    
    
            
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()

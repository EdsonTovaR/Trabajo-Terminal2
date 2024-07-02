import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.cluster import KMeans

class DataCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de Limpieza de Datos con Árboles de Decisiones")
        self.data = None
        
        # Elementos de la interfaz
        self.create_widgets()
    
    def create_widgets(self):
        self.load_button = tk.Button(self.root, text="Cargar Archivo CSV", command=self.load_file)
        self.load_button.pack(pady=10)
        
        self.column_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.column_listbox.pack(pady=10)
        
        self.strategy_var = tk.StringVar(value="mean")
        self.strategy_menu = tk.OptionMenu(self.root, self.strategy_var, "mean", "median", "most_frequent", "KNN", "Iterative")
        self.strategy_menu.pack(pady=10)
        
        self.impute_button = tk.Button(self.root, text="Imputar Datos", command=self.impute_data)
        self.impute_button.pack(pady=10)
        
        self.outlier_button = tk.Button(self.root, text="Detectar Outliers", command=self.detect_outliers)
        self.outlier_button.pack(pady=10)
        
        self.suggest_button = tk.Button(self.root, text="Sugerir Transformaciones", command=self.suggest_transformations)
        self.suggest_button.pack(pady=10)
        
        self.target_menu = tk.Listbox(self.root, selectmode=tk.SINGLE)
        self.target_menu.pack(pady=10)
        
        self.features_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.features_listbox.pack(pady=10)
        
        self.train_button = tk.Button(self.root, text="Entrenar Árbol de Decisiones", command=self.train_model)
        self.train_button.pack(pady=10)
        
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(pady=10)
    
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.update_column_listbox()
            messagebox.showinfo("Carga de Datos", "Archivo cargado exitosamente.")
    
    def update_column_listbox(self):
        self.column_listbox.delete(0, tk.END)
        self.target_menu.delete(0, tk.END)
        self.features_listbox.delete(0, tk.END)
        
        for col in self.data.columns:
            self.column_listbox.insert(tk.END, col)
            self.target_menu.insert(tk.END, col)
            self.features_listbox.insert(tk.END, col)
    
    def impute_data(self):
        selected_indices = self.column_listbox.curselection()
        selected_columns = [self.column_listbox.get(i) for i in selected_indices]
        
        if selected_columns:
            strategy = self.strategy_var.get()
            
            if strategy == "mean" or strategy == "median" or strategy == "most_frequent":
                imputer = SimpleImputer(strategy=strategy)
            elif strategy == "KNN":
                imputer = KNNImputer()
            elif strategy == "Iterative":
                imputer = IterativeImputer()
            else:
                messagebox.showwarning("Imputación", "Estrategia de imputación no válida.")
                return
            
            self.data[selected_columns] = imputer.fit_transform(self.data[selected_columns])
            messagebox.showinfo("Imputación", "Datos imputados exitosamente.")
        else:
            messagebox.showwarning("Imputación", "Seleccione columnas para imputar.")
    
    def detect_outliers(self):
        selected_indices = self.column_listbox.curselection()
        selected_columns = [self.column_listbox.get(i) for i in selected_indices]
        
        if selected_columns:
            detector = IsolationForest(contamination=0.1)
            outliers = detector.fit_predict(self.data[selected_columns])
            outlier_indices = self.data[outliers == -1].index
            self.data = self.data.drop(outlier_indices)
            messagebox.showinfo("Detección de Outliers", f"Se eliminaron {len(outlier_indices)} outliers.")
        else:
            messagebox.showwarning("Detección de Outliers", "Seleccione columnas para la detección de outliers.")
    
    def suggest_transformations(self):
        selected_indices = self.column_listbox.curselection()
        selected_columns = [self.column_listbox.get(i) for i in selected_indices]
        
        if selected_columns:
            kmeans = KMeans(n_clusters=2)
            clusters = kmeans.fit_predict(self.data[selected_columns])
            
            # Aquí podrías agregar lógica para sugerir transformaciones basadas en los clusters
            messagebox.showinfo("Sugerencias", "Transformaciones sugeridas basadas en clusters.")
        else:
            messagebox.showwarning("Sugerencias", "Seleccione columnas para las sugerencias.")
    
    def train_model(self):
        target_index = self.target_menu.curselection()
        feature_indices = self.features_listbox.curselection()
        
        if target_index and feature_indices:
            target_column = self.target_menu.get(target_index)
            feature_columns = [self.features_listbox.get(i) for i in feature_indices]
            
            X = self.data[feature_columns]
            y = self.data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            messagebox.showinfo("Resultados", f"Precisión del modelo: {accuracy:.2f}")
            
            # Visualización del árbol
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(model, filled=True, feature_names=feature_columns, class_names=target_column, ax=ax)
            
            # Limpiar el frame de la visualización previa
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            messagebox.showwarning("Entrenamiento", "Seleccione una columna objetivo y características.")
    
def main():
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

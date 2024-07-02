import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd

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
        
        #Boton para mostrar datos
        self.show_button = tk.Button(root, text="Mostrar Datos", command=self.mostrar_datos, bg="blue", fg="white", font=("century gothic", 14))
        self.show_button.pack(padx=20, pady=20)
        self.show_button.config(state=tk.DISABLED)
        
        #Texto para mostrar los datos
        self.data_text = tk.Text(root, width=100, height=10, bg="white", fg="black", font=("century gothic", 12))
        self.data_text.pack(padx=20, pady=20)
        self.data_text.config(state=tk.DISABLED)
        
        #Boton para seleccionar las opciones de limpieza    
        self.cleaning_button = tk.Button(root, text="Opciones de Limpieza", command=self.limpiar_datos, bg="blue", fg="white", font=("century gothic", 14))
        self.cleaning_button.pack(padx=20, pady=20)
        
       
        
        
        
        
    def cargar_datos(self, event):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.data = pd.read_csv(file_path)
                self.label.config(text="Archivo cargado: " + file_path.split("/")[-1])
                self.clean_button.config(state=tk.NORMAL)
                datos=self.data 
                print(self.data.head)
                self.show_button.config(state=tk.NORMAL)
        except Exception as e:
            self.label.config(text=f"Error al cargar el archivo: {str(e)}")
            self.clean_button.config(state=tk.DISABLED)
            
    def guardar_datos(self):
        try:
            self.data.to_csv("cleaned_data.csv", index=False)
            self.label.config(text="Datos guardados como 'cleaned_data.csv'")
        except Exception as e:
            self.label.config(text=f"Error al guardar el archivo: {str(e)}")
            
    #Mostramos los datos en la interfaz con un resumen de ellos
    def mostrar_datos(self):
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete("1.0", tk.END)
        self.data_text.insert(tk.END, f"Resumen de los datos:\n\n")
        self.data_text.insert(tk.END, f"Columnas: {self.data.shape[1]}\n")
        self.data_text.insert(tk.END, f"Filas: {self.data.shape[0]}\n\n")
        self.data_text.insert(tk.END, f"{self.data.head()}")
        self.data_text.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.confidence_button.config(state=tk.NORMAL)
        
    
    
        
    def limpiar_datos(self):
        return
            
   
  
  
  
  
    

        
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()
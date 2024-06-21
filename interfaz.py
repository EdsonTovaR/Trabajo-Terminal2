import tkinter as tk
import pandas as pd
from tkinter import Tk

# Creamos la interfaz
root = Tk()
root.title("Interfaz de usuario")
root.geometry("400x400")

# Creamos la variable para almacenar la ruta del archivo CSV
ruta_archivo = tk.StringVar()

# Creamos la función para importar el archivo CSV
def importar():
  """
  Función para importar el archivo CSV y mostrar su contenido en la interfaz.
  """
  # Obtenemos la ruta del archivo seleccionado
  ruta_seleccionada = tk.filedialog.askopenfilename(
      title="Seleccionar archivo CSV",
      filetypes=[("Archivos CSV", "*.csv")])

  # Si el usuario selecciona un archivo, continuamos
  if ruta_seleccionada:
    # Actualizamos la variable con la ruta del archivo seleccionado
    ruta_archivo.set(ruta_seleccionada)

    # Leemos el archivo CSV en un DataFrame de Pandas
    try:
      df = pd.read_csv(ruta_seleccionada)

      # Mostramos el contenido del DataFrame en un Treeview
      treeview = tk.ttk.Treeview(root)
      treeview.pack()

      # Configuramos las columnas del Treeview
      columnas = df.columns
      treeview["columns"] = columnas

      # Agregamos los encabezados de las columnas
      for columna in columnas:
        treeview.heading(columna, text=columna)

      # Insertamos los datos del DataFrame en el Treeview
      datos = df.values
      for fila in datos:
        treeview.insert("", "end", values=fila)

    except Exception as e:
      # En caso de error, mostramos un mensaje
      tk.messagebox.showerror("Error", f"Error al leer el archivo: {e}")

# Creamos una etiqueta para mostrar la ruta del archivo seleccionado
etiqueta_ruta = tk.Label(root, textvariable=ruta_archivo)
etiqueta_ruta.pack()

# Creamos un botón para importar el archivo
boton = tk.Button(root, text="Importar archivo", command=importar)
boton.pack()

root.mainloop()
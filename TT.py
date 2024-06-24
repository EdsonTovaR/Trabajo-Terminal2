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

    def preprocess_data(self, X):
        # Detectar y preprocesar características numéricas y categóricas
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Ignorar las columnas que no se transforman
        )
        
        return preprocessor

    def clean_data(self):
        if self.data is not None:
            self.progress.start()
            self.impute_missing_values()
            self.progress.stop()
            self.label.config(text="Datos limpiados.")
            self.save_button.config(state=tk.NORMAL)

    def save_data(self):
        try:
            self.data.to_csv("cleaned_data.csv", index=False)
            self.label.config(text="Datos guardados como 'cleaned_data.csv'")
        except Exception as e:
            self.label.config(text=f"Error al guardar el archivo: {str(e)}")

    def impute_missing_values(self):
        for column in self.data.columns:
            if self.data[column].isnull().any():
                if self.evaluate_imputation(column) <= 0.1:  # Threshold for acceptable MSE
                    self.impute_column(column)
                else:
                    print(f"Column {column} has high MSE for imputation. Skipping.")

    def impute_column(self, column):
        df = self.data.copy()
        train = df[df[column].notnull()]
        test = df[df[column].isnull()]

        if not test.empty:
            X_train = train.drop(columns=[column])
            y_train = train[column]
            X_test = test.drop(columns=[column])

            preprocessor = self.preprocess_data(X_train)
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
            tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=3)  # Cambiado a cv=3 para tiempos más cortos
            tree.fit(X_train, y_train)

            predicted_values = tree.predict(X_test)
            self.data.loc[self.data[column].isnull(), column] = predicted_values

    def evaluate_imputation(self, column):
        df = self.data.copy()
        df_known = df[df[column].notnull()]
        df_unknown = df[df[column].isnull()]

        if not df_unknown.empty:
            X_train = df_known.drop(columns=[column])
            y_train = df_known[column]

            preprocessor = self.preprocess_data(X_train)
            X_train = preprocessor.fit_transform(X_train)

            param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
            tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=3)  # Cambiado a cv=3 para tiempos más cortos
            tree.fit(X_train, y_train)

            X_test = X_train
            y_pred = tree.predict(X_test)
            
            mse = mean_squared_error(y_train, y_pred)
            print(f'Mean Squared Error for column {column}: {mse}')
            return mse  # Return the MSE value
if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanerApp(root)
    root.mainloop()

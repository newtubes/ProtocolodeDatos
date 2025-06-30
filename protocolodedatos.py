import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

# Clase principal para el procesamiento de datasets
# Esta clase encapsula las funcionalidades de carga, limpieza y preprocesamiento de datos.
class DatasetProcessor:
    """
    Una clase para automatizar la recolección, limpieza y preprocesamiento de datos
    para el entrenamiento de modelos de Machine Learning.
    """

    def __init__(self, data=None, file_path=None):
        """
        Inicializa el procesador de datasets.
        Puede cargar datos directamente desde un DataFrame de pandas o desde un archivo.

        Args:
            data (pd.DataFrame, opcional): Un DataFrame de pandas existente.
            file_path (str, opcional): La ruta al archivo de datos (CSV o JSON).
        """
        self.df = None
        if data is not None:
            self.df = data
            print("Datos cargados directamente en el procesador.")
        elif file_path:
            self.load_data(file_path)
        else:
            print("Inicializado sin datos. Usa 'load_data()' para cargar un dataset.")

    def load_data(self, file_path):
        """
        Carga datos desde un archivo CSV o JSON.

        Args:
            file_path (str): La ruta al archivo de datos.
        """
        if not os.path.exists(file_path):
            print(f"Error: El archivo no se encontró en la ruta: {file_path}")
            return

        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
                print(f"Datos cargados exitosamente desde CSV: {file_path}")
            elif file_path.endswith('.json'):
                self.df = pd.read_json(file_path)
                print(f"Datos cargados exitosamente desde JSON: {file_path}")
            else:
                print("Formato de archivo no soportado. Por favor, usa .csv o .json.")
                self.df = None
        except Exception as e:
            print(f"Error al cargar los datos desde {file_path}: {e}")
            self.df = None

    def display_info(self):
        """
        Muestra información básica sobre el DataFrame actual (si existe).
        Incluye las primeras filas, tipos de datos y estadísticas descriptivas.
        """
        if self.df is not None:
            print("\n--- Información del Dataset ---")
            print("Primeras 5 filas:")
            print(self.df.head())
            print("\nInformación general (tipos de datos, valores no nulos):")
            self.df.info()
            print("\nEstadísticas descriptivas:")
            print(self.df.describe(include='all'))
            print("\nConteo de valores nulos por columna:")
            print(self.df.isnull().sum())
        else:
            print("No hay datos cargados para mostrar información.")

    def handle_missing_values(self, strategy='mean', columns=None, fill_value=None):
        """
        Maneja los valores nulos en el DataFrame.

        Args:
            strategy (str): Estrategia para manejar los nulos.
                            'mean': Rellena con la media (solo numéricas).
                            'median': Rellena con la mediana (solo numéricas).
                            'mode': Rellena con la moda (numéricas y categóricas).
                            'ffill': Rellena hacia adelante (forward fill).
                            'bfill': Rellena hacia atrás (backward fill).
                            'drop': Elimina las filas con valores nulos.
                            'constant': Rellena con un valor constante especificado por 'fill_value'.
            columns (list, opcional): Lista de columnas a procesar. Si es None, procesa todas las columnas.
            fill_value (any, opcional): Valor a usar si la estrategia es 'constant'.
        """
        if self.df is None:
            print("No hay datos cargados para manejar valores nulos.")
            return

        cols_to_process = columns if columns is not None else self.df.columns.tolist()
        print(f"\nManejando valores nulos con estrategia '{strategy}' en columnas: {cols_to_process}")

        for col in cols_to_process:
            if col not in self.df.columns:
                print(f"Advertencia: La columna '{col}' no existe en el DataFrame. Saltando.")
                continue

            if self.df[col].isnull().sum() == 0:
                print(f"No hay valores nulos en la columna '{col}'.")
                continue

            if strategy == 'mean':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    print(f"Columna '{col}': nulos rellenados con la media.")
                else:
                    print(f"Advertencia: La columna '{col}' no es numérica. No se puede usar la estrategia 'mean'.")
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"Columna '{col}': nulos rellenados con la mediana.")
                else:
                    print(f"Advertencia: La columna '{col}' no es numérica. No se puede usar la estrategia 'median'.")
            elif strategy == 'mode':
                # La moda puede devolver múltiples valores, tomamos el primero
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                print(f"Columna '{col}': nulos rellenados con la moda ({mode_val}).")
            elif strategy == 'ffill':
                self.df[col].fillna(method='ffill', inplace=True)
                print(f"Columna '{col}': nulos rellenados con forward fill.")
            elif strategy == 'bfill':
                self.df[col].fillna(method='bfill', inplace=True)
                print(f"Columna '{col}': nulos rellenados con backward fill.")
            elif strategy == 'drop':
                initial_rows = self.df.shape[0]
                self.df.dropna(subset=[col], inplace=True)
                print(f"Columna '{col}': filas con nulos eliminadas. Filas eliminadas: {initial_rows - self.df.shape[0]}")
            elif strategy == 'constant':
                if fill_value is not None:
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Columna '{col}': nulos rellenados con el valor constante '{fill_value}'.")
                else:
                    print("Error: Para la estrategia 'constant', 'fill_value' debe ser especificado.")
            else:
                print(f"Estrategia '{strategy}' no reconocida para la columna '{col}'.")
        
        # Después de procesar columnas específicas, si aún quedan nulos en otras, se reporta.
        if self.df.isnull().sum().sum() > 0:
            print("\nConteo de valores nulos restantes después del procesamiento:")
            print(self.df.isnull().sum())
        else:
            print("\nTodos los valores nulos han sido manejados.")


    def remove_duplicates(self, subset=None, keep='first'):
        """
        Elimina filas duplicadas del DataFrame.

        Args:
            subset (list, opcional): Lista de nombres de columnas para considerar al identificar duplicados.
                                     Si es None, considera todas las columnas.
            keep (str): Determina qué duplicados conservar.
                        'first': Conserva la primera ocurrencia.
                        'last': Conserva la última ocurrencia.
                        False: Elimina todas las ocurrencias duplicadas.
        """
        if self.df is None:
            print("No hay datos cargados para eliminar duplicados.")
            return

        initial_rows = self.df.shape[0]
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        rows_removed = initial_rows - self.df.shape[0]
        print(f"\nDuplicados eliminados. Filas eliminadas: {rows_removed}")
        if rows_removed > 0:
            print(f"Filas restantes: {self.df.shape[0]}")
        else:
            print("No se encontraron duplicados para eliminar.")

    def encode_categorical(self, columns, encoder_type='onehot'):
        """
        Codifica variables categóricas.

        Args:
            columns (list): Lista de nombres de columnas a codificar.
            encoder_type (str): Tipo de codificación. Actualmente solo 'onehot' soportado.
        """
        if self.df is None:
            print("No hay datos cargados para codificar variables categóricas.")
            return

        if encoder_type == 'onehot':
            print(f"\nRealizando One-Hot Encoding en columnas: {columns}")
            for col in columns:
                if col not in self.df.columns:
                    print(f"Advertencia: La columna '{col}' no existe en el DataFrame. Saltando.")
                    continue
                if not pd.api.types.is_object_dtype(self.df[col]) and not pd.api.types.is_categorical_dtype(self.df[col]):
                    print(f"Advertencia: La columna '{col}' no parece ser categórica. Saltando codificación.")
                    continue

                try:
                    # Inicializa OneHotEncoder
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    # Ajusta y transforma la columna
                    encoded_data = encoder.fit_transform(self.df[[col]])
                    # Crea un DataFrame con los nombres de las nuevas columnas
                    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]), index=self.df.index)
                    # Concatena el DataFrame codificado con el original y elimina la columna original
                    self.df = pd.concat([self.df.drop(columns=[col]), encoded_df], axis=1)
                    print(f"Columna '{col}' codificada exitosamente.")
                except Exception as e:
                    print(f"Error al codificar la columna '{col}': {e}")
        else:
            print(f"Tipo de codificador '{encoder_type}' no soportado.")

    def get_processed_data(self):
        """
        Retorna el DataFrame procesado.

        Returns:
            pd.DataFrame: El DataFrame actual.
        """
        return self.df

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Crear un archivo CSV de ejemplo para probar
    sample_data_csv = {
        'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Edad': [25, 30, np.nan, 35, 25, 40, 30, np.nan, 28, 32],
        'Ciudad': ['Buenos Aires', 'Córdoba', 'Rosario', 'Buenos Aires', 'Córdoba', 'Rosario', 'Buenos Aires', 'Córdoba', 'Rosario', 'Buenos Aires'],
        'Ingresos': [50000, 60000, 45000, 70000, np.nan, 80000, 55000, 62000, 48000, 75000],
        'Genero': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
        'Experiencia': [2, 5, 1, 8, 3, 10, 4, 6, 2, 7]
    }
    df_sample_csv = pd.DataFrame(sample_data_csv)
    # Añadir algunos duplicados y nulos adicionales para probar la limpieza
    df_sample_csv.loc[10] = [1, 25, 'Buenos Aires', 50000, 'F', 2] # Duplicado completo
    df_sample_csv.loc[11] = [11, 30, 'Córdoba', np.nan, 'M', 5] # Nulo en ingresos
    df_sample_csv.loc[12] = [12, np.nan, 'Rosario', 45000, 'F', 1] # Nulo en edad
    df_sample_csv.loc[13] = [1, 25, 'Buenos Aires', 50000, 'F', 2] # Otro duplicado completo
    
    csv_file_path = 'sample_data.csv'
    df_sample_csv.to_csv(csv_file_path, index=False)
    print(f"Archivo de ejemplo '{csv_file_path}' creado.")

    # Crear un archivo JSON de ejemplo para probar
    sample_data_json = [
        {"product_id": "A001", "name": "Laptop", "category": "Electronics", "price": 1200.00, "stock": 50},
        {"product_id": "A002", "name": "Mouse", "category": "Electronics", "price": 25.00, "stock": np.nan},
        {"product_id": "B001", "name": "Keyboard", "category": "Electronics", "price": 75.00, "stock": 100},
        {"product_id": "C001", "name": "Desk Lamp", "category": "Home", "price": 40.00, "stock": 20},
        {"product_id": "A001", "name": "Laptop", "category": "Electronics", "price": 1200.00, "stock": 50}, # Duplicado
        {"product_id": "D001", "name": "Coffee Maker", "category": "Kitchen", "price": np.nan, "stock": 30}
    ]
    json_file_path = 'sample_data.json'
    pd.DataFrame(sample_data_json).to_json(json_file_path, orient='records', indent=4)
    print(f"Archivo de ejemplo '{json_file_path}' creado.")

    # --- Flujo de trabajo 1: Cargar CSV, limpiar y preprocesar ---
    print("\n--- Flujo de Trabajo 1: Procesando sample_data.csv ---")
    processor_csv = DatasetProcessor(file_path=csv_file_path)
    processor_csv.display_info()

    # Manejar valores nulos: rellenar 'Edad' con la mediana, 'Ingresos' con la media
    processor_csv.handle_missing_values(strategy='median', columns=['Edad'])
    processor_csv.handle_missing_values(strategy='mean', columns=['Ingresos'])
    # Rellenar nulos restantes (si los hubiera) en columnas categóricas con la moda
    processor_csv.handle_missing_values(strategy='mode', columns=['Ciudad', 'Genero'])

    processor_csv.display_info() # Verificar después de rellenar nulos

    # Eliminar filas duplicadas
    processor_csv.remove_duplicates()

    processor_csv.display_info() # Verificar después de eliminar duplicados

    # Codificar variables categóricas
    processor_csv.encode_categorical(columns=['Ciudad', 'Genero'])

    print("\n--- Dataset CSV Procesado Final ---")
    print(processor_csv.get_processed_data().head())
    print(processor_csv.get_processed_data().info())


    # --- Flujo de trabajo 2: Cargar JSON, limpiar y preprocesar ---
    print("\n\n--- Flujo de Trabajo 2: Procesando sample_data.json ---")
    processor_json = DatasetProcessor(file_path=json_file_path)
    processor_json.display_info()

    # Manejar valores nulos: rellenar 'stock' con la media, 'price' con un valor constante
    processor_json.handle_missing_values(strategy='mean', columns=['stock'])
    processor_json.handle_missing_values(strategy='constant', columns=['price'], fill_value=0.0)

    processor_json.display_info() # Verificar después de rellenar nulos

    # Eliminar filas duplicadas
    processor_json.remove_duplicates()

    processor_json.display_info() # Verificar después de eliminar duplicados

    # Codificar variables categóricas
    processor_json.encode_categorical(columns=['category'])

    print("\n--- Dataset JSON Procesado Final ---")
    print(processor_json.get_processed_data().head())
    print(processor_json.get_processed_data().info())

    # Limpiar archivos de ejemplo
    os.remove(csv_file_path)
    os.remove(json_file_path)
    print(f"\nArchivos de ejemplo '{csv_file_path}' y '{json_file_path}' eliminados.")

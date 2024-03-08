import pandas as pd

# Ruta del archivo CSV
csv_file = "homeLoanAproval.csv"

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(csv_file)

# Eliminar las filas que contienen datos faltantes
df.dropna(inplace=True)

# Eliminar las filas que tengan un '3+'
df = df[df['Dependents'] != '3+']

# Pasamos a string la columna de números
df['ApplicantIncome'] = df['ApplicantIncome'].astype(str)

# Quitamos los números mal escritos
df = df[~df['ApplicantIncome'].str.contains('\.')]

# Pasamos la columna a números 
df['ApplicantIncome'] = df['ApplicantIncome'].astype(int)

# Guardar el DataFrame actualizado en un nuevo archivo CSV
df.to_csv("homeLoanAproval_modificado.csv", index=False)
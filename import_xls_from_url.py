import pandas as pd
import os

# Chemin du dossier contenant les fichiers CSV
folder_path = rf"/Users/Papa/Desktop/EuroMillions"

# Liste pour stocker les DataFrames
dataframes = []

# Parcourir tous les fichiers CSV dans le dossier
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Vérifie que le fichier a l'extension .csv
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)  # Lire le fichier CSV dans un DataFrame
        dataframes.append(df)  # Ajouter le DataFrame à la liste

# Combiner tous les DataFrames en un seul
combined_df = pd.concat(dataframes, ignore_index=True)

# Exporter le DataFrame combiné dans un fichier CSV unique
output_file = os.path.join(folder_path, "euromillions.csv")
combined_df.to_csv(output_file, index=False)

print(f"Fichier CSV combiné créé : {output_file}")
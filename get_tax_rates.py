import pandas as pd

def read_vat_data():
    file_path = "dataset\VAT Rates in Europe  EU VAT Rates by Country.csv"
    df = pd.read_csv(file_path)
    return df

data = read_vat_data()
print("Data shape:", data.shape)
print("Columns:", list(data.columns))
print("First 5 rows:")
print(data.head())

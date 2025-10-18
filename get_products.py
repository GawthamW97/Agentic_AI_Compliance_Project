import pandas as pd

def read_flipkart_data():
    file_path = "dataset/flipkart_com-ecommerce_sample.csv"
    df = pd.read_csv(file_path)
    return df

data = read_flipkart_data()
print("Data shape:", data.shape)
print("Columns:", list(data.columns))

import pandas as pd
import requests
import os
import csv
from dotenv import load_dotenv

CODE = "1001000000"

BASEURL = "https://tulltaxan.tullverket.se/ite-tariff-public-proxy/ite-tariff-trusted-rs/v1/sbn/nomenclatures/"
def readCNCodes():
    cnCodes_df = pd.read_csv("dataset/tax_codes.csv")
    # Remove spaces from tax_code
    cnCodes_df["tax_code"] = cnCodes_df["tax_code"].str.replace(' ', '')
    cnCodes_df["tax_code"] = cnCodes_df["tax_code"].str.rstrip('00')
    # Filter rows where tax_code length is 8
    return cnCodes_df[cnCodes_df["tax_code"].str.len() == 8].sort_values(by="tax_code")

df = readCNCodes()
df.head()

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Site": "same-origin",
    "Origin": "https://tulltaxan.tullverket.se",
    "Referer": "https://tulltaxan.tullverket.se/ite-tariff-public/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors"
}

os.makedirs('dataset', exist_ok=True)

fieldnames = ["Code","Description","Declarable"]

current_batch = []
total_records = 0
batch_size = 1000
batch_num = 1
for code in df["tax_code"]:
    URL =  BASEURL + code
    params = {"language":"en","simulationdate":"2025-10-16","tradedirection":"I"}
    try:
        response = requests.get(URL,params=params,headers=headers)
        if response.status_code == 200:
            taric_codes = response.json()
            if len(taric_codes["childItems"]) != 0:
                print(code)
                for item in taric_codes["childItems"]:
                    if(len(item["item"]["code"]) == 10):
                        struct = {"Code":item["item"]["code"],"Description":item["item"]["description"],"Declarable":item["item"]["declarable"]}
                        current_batch.append(struct)
                        if len(current_batch) == batch_size:
                            batch_file = os.path.join('dataset', f'taric_codes_batch_{batch_num}.csv')
                            with open(batch_file, 'w', newline='', encoding='utf-8') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(current_batch)
                            print(f"Batch {batch_num} saved to {batch_file}")
                            current_batch = []
                            total_records += batch_size
                            batch_num += 1
        else:
            print(response.content)
    except:
            print(response.content)
            break

print("Retrieved all TARIC codes!!!")

if current_batch:
    batch_file = os.path.join('dataset', f'taric_codes_batch_{batch_num}.csv')
    with open(batch_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(current_batch)
    print(f"Batch {batch_num} saved to {batch_file}")
    total_records += len(current_batch)

print(f"All batches saved to dataset/ directory")
print(f"Total records: {total_records}")

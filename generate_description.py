from openai import OpenAI
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

hs_df = pd.read_csv("./dataset/harmonized-system.csv")
hs_df["hscode"] = hs_df["hscode"].astype(str)
hs_df = hs_df[hs_df["hscode"].str.len() >= 5]
hs_df["hscode"] = hs_df["hscode"].astype("int64")
hs_df = hs_df.sample(n=100, random_state=42)

records = []

for _, row in hs_df.iterrows():
    prompt = f"""
    You are a data generation assistant creating realistic product descriptions for international trade datasets.

    Given the following HS code and description:
    HS Code: {row['hscode']}
    HS Description: {row['description']}

    Generate 3 different product descriptions that might appear in real sales or import/export datasets.
    Each description must:
    - Be written in natural commercial language.
    - Avoid repeating the HS description verbatim.
    - Reflect real-world variation (brand-style phrasing, product types, short/long formats).
    - Contain between 8 and 25 words.
    - Optionally include adjectives, quantity, or intended use.
    - Avoid fictional brands or unrealistic product names.

    Output format:
    1. <description>
    2. <description>
    3. <description>
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )

    content = response.choices[0].message.content.strip()
    lines = content.split("\n")

    for line in lines:
        if line.strip() and line[0].isdigit():
            desc = line.split(".", 1)[1].strip()
            records.append({"hscode": row["hscode"], "description": desc})

    time.sleep(0.5)

df = pd.DataFrame(records)
df.to_csv("synthetic_product_descriptions.csv", index=False)
print("Synthetic product descriptions saved to synthetic_product_descriptions.csv")

import requests
from bs4 import BeautifulSoup
import ftfy
import re
import os
import csv
import unicodedata
import html
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv('CN_TAX_CODES')

# Regex for CN/TARIC codes like "2517 20 00"
TAX_CODE_PATTERN = re.compile(r"^\d{4}(?:\s\d{2}){0,2}$")

def scrape_products(url):
    """
    Scrape tax codes and descriptions from the given URL.
    Handles encoding issues and returns clean UTF-8 data.
    """
    # Request with explicit encoding handling
    resp = requests.get(url)
    resp.raise_for_status()
    
    # Detect and set proper encoding
    resp.encoding = resp.apparent_encoding or 'utf-8'
    
    # Parse with explicit encoding
    soup = BeautifulSoup(resp.content, "html.parser", from_encoding=resp.encoding)
    
    products = []
    
    for row in soup.find_all("tr"):
        cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cols) >= 2:
            first_col = cols[0]
            if TAX_CODE_PATTERN.match(first_col):
                tax_code = first_col.replace('\xa0', ' ').strip()

                # Join description and fix encoding issues
                raw_description = " ".join(cols[1:])
                # Fix mojibake and other encoding problems
                clean_description = ftfy.fix_text(raw_description)
                # Normalize Unicode to remove artifacts
                clean_description = unicodedata.normalize('NFKC', clean_description)
                # Unescape HTML entities if any
                clean_description = html.unescape(clean_description)
                # Remove non-printable characters
                clean_description = ''.join(c for c in clean_description if c.isprintable())
                # Remove unnecessary characters like en dashes and normalize spaces
                clean_description = clean_description.replace('–', '').strip()
                clean_description = re.sub(r'\s+', ' ', clean_description)
                
                products.append({
                    "tax_code": tax_code,
                    "description": clean_description
                })
    
    return products

# Scrape the data
print(f"Scraping data from {URL}...")
data = scrape_products(URL)
print(f"Found {len(data)} tax codes")

# Save to CSV with proper UTF-8 encoding
output_path = os.path.join('dataset', 'tax_codes.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['tax_code', 'description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)

print(f"Data saved to {output_path}")
print(f"Total records: {len(data)}")
import requests
from xml.etree import ElementInclude as ET

def get_taric_measures(code):
    url = 'https://tulltaxan.tullverket.se/ite-tariff-public-proxy/ite-tariff-trusted-rs/v1/mcc/measures'
    params = {
        'count': '12',
        'offset': '0',
        'sortorder': 'A',
        'simulationdate': '2025-10-14',
        'tradedirection': 'I',
        'commoditycode': code,
        'currency': 'EUR'
    }
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://tulltaxan.tullverket.se',
        'Referer': 'https://tulltaxan.tullverket.se/ite-tariff-public/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Cookie': 'RASZOUPN=02dc599f28-7428-41vhmzBS_gwOG_goDvJIV8ggNwBPEv_TXZy8o2nnWM6h-HyMmdLnLeHKl38nPAbPrlDOI'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching TARIC measures: {e}")
        return None


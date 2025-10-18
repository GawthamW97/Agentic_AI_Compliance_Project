from dataclasses import dataclass

@dataclass
class Product:
    product_id: str
    description: str
    hs_code: str
    cn_code: str
    origin_country: str
    destination_country: str
    measure_data_id: int

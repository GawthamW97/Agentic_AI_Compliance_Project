from dataclasses import dataclass
from typing import Optional

@dataclass
class ProductMeasure:
    measure_id: str
    measure_type: str
    rate: str
    unit: str
    validity_period: str
    additional_code: str
    regulation_ref: str
    remarks: Optional[str] = None

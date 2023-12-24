from dataclasses import dataclass
from datetime import date, datetime
from json import JSONEncoder

@dataclass
class TransactionModel():
    name: str
    owner: str
    asset_description: str
    asset_type: str
    type: str
    amount: float
    comment: str
    created_at: date
    price: int

class TransactionEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TransactionModel):
            # Convert date to string
            obj.created_at = obj.created_at.isoformat() if obj.created_at else None
            return obj.__dict__
        return super().default(obj)

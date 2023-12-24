from dataclasses import dataclass
from json import JSONEncoder

@dataclass
class TransactionModel():
    owner: str
    asset_description: str
    asset_type: str
    type: str
    amount: float
    comment: str

class TransactionEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TransactionModel):
            return obj.__dict__
        return super().default(obj)
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from json import JSONEncoder
from typing import TYPE_CHECKING, TypedDict, Union


# To avoid creating a class at runtime, for type-hinting alone.
if TYPE_CHECKING:
    class ErrorEnumType(TypedDict):
        SYMBOL_DELISTED: int
        DATA_DOES_NOT_EXIST: int
        DATA_NOT_FOUND: int


ErrorEnum: ErrorEnumType = {
    'SYMBOL_DELISTED': -1,
    # if the API return error
    "DATA_DOES_NOT_EXIST": -2,
    # If the API doesn't return error or data
    "DATA_NOT_FOUND": -3,
}

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
    price1d: int
    price7d: int
    price15d: int
    price1m: int
    price6m: int
    price1y: int

class TransactionEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TransactionModel):
            # Convert date to string
            obj.created_at = obj.created_at.isoformat() if obj.created_at else None
            return obj.__dict__
        return super().default(obj)

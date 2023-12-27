from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Union


# To avoid creating a class at runtime, for type-hinting alone.
if TYPE_CHECKING:
    class ErrorEnumType(TypedDict):
        SYMBOL_DELISTED: int
        DATA_DOES_NOT_EXIST: int
        DATA_NOT_FOUND: int


ErrorEnumModel: ErrorEnumType = {
    'SYMBOL_DELISTED': -1,
    # if the API return error
    "DATA_DOES_NOT_EXIST": -2,
    # If the API doesn't return error or data
    "DATA_NOT_FOUND": -3,
}
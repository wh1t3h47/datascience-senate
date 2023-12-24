from json import load, dump
from os.path import join
from os import listdir
from typing import List, Union

from ..models.TransactionModel import TransactionEncoder, TransactionModel



def normalize_data(data: List[List[TransactionModel]]) -> List[TransactionModel]:
    """
    Remove transações com type stock de zero.

    Args:
        data: Lista de listas de objetos TransactionModel.

    Returns:
        Lista de objetos TransactionModel normalizados.
    """
    return [transaction for senator_transactions in data for transaction in senator_transactions if transaction.asset_type == "Stock"]

def get_transactions(path: str) -> Union[List[List[TransactionModel]], None]:
    """
    Retorna uma lista de listas de objetos TransactionModel a partir de um caminho de arquivo.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Lista de listas de objetos TransactionModel.
    """
    if not path.endswith('.json'):
        return None
    
    with open(join('.', 'data', path), "r") as f:
        data = load(f)

    transactions = [
        [
            TransactionModel(
                owner=(
                    senator.get("first_name", "") + " " + senator.get("last_name", "")
                    if t.get("owner", "") == "Self"
                    else t.get("owner", "")
                ),
                asset_description=t.get("asset_description", ""),
                asset_type=t.get("asset_type", ""),
                type=t.get("type", ""),
                amount=t.get("amount", 0.0),
                comment=t.get("comment", "")
            )
            for t in senator.get("transactions", [])
        ]
        for senator in data or []
    ]
    return transactions if transactions else None

def main():
    path = join(".", "data")

    transactions = [get_transactions(f) for f in listdir(path) if f.endswith('.json')]
    print('1', transactions)
    transactions = normalize_data([transaction for sublist in transactions if sublist for transaction in sublist])
    print('2', transactions)
    transactions_json = {"transactions": transactions}
    print('3', transactions_json)

    with open("transactions.json", "w") as f:
        dump(transactions_json, f, cls=TransactionEncoder)

if __name__ == "__main__":
    main()

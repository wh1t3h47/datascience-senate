from json import load, dump
from os.path import join
from os import listdir
from typing import List, Union
from datetime import datetime

from ..services.stocks_api_service import stocks_api_service

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

def get_transactions() -> Union[List[List[TransactionModel]], None]:
    """
    Retorna uma lista de listas de objetos TransactionModel a partir de um caminho de arquivo.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Lista de listas de objetos TransactionModel.
    """
    data = {}
    path = join(".", "aggregate", 'all_transactions_for_senators.json')
    with open(path, 'r') as _data:
        data = load(_data)

    transactions = [
        [
            TransactionModel(
                name=(
                    senator.get("first_name", "") + " " + senator.get("last_name", "")
                ),
                owner=(
                    t.get('owner', '')
                ),
                asset_description=t.get("asset_description", ""),
                asset_type=t.get("asset_type", ""),
                type=t.get("type", ""),
                amount=t.get("amount", 0.0),
                comment=t.get("comment", ""),
                created_at=datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(),
                # @todo handle tick fail (when tick doesn't exist)
                price=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), ticker=t.get("ticker", "")),
            )
            for t in senator.get("transactions", [])
        ]
        for senator in data or []
    ]
    return transactions if transactions else None

def main():
    transactions = get_transactions()
    transactions = normalize_data([transaction for sublist in transactions if sublist for transaction in sublist])
    transactions_json = {"transactions": transactions}

    with open("transactions.json", "w") as f:
        dump(transactions_json, f, cls=TransactionEncoder)

if __name__ == "__main__":
    main()

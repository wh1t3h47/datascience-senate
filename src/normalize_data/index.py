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
    return [transaction for senator_transactions in data if senator_transactions for transaction in senator_transactions if transaction.asset_type == "Stock"]

def get_transactions() -> Union[List[List[TransactionModel]], None]:
    """
    Retorna uma lista de listas de objetos TransactionModel

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
                price_avarage_in_1d=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="1d", ticker=t.get("ticker", ""), type="avg"),
                price_avarage_in_7d=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="7d", ticker=t.get("ticker", ""), type="avg"),
                price_avarage_in_15d=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="15d", ticker=t.get("ticker", ""), type="avg"),
                price_avarage_in_1m=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="1m", ticker=t.get("ticker", ""), type="avg"),
                price_avarage_in_6m=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="6m", ticker=t.get("ticker", ""), type="avg"),
                price_avarage_in_1y=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="1y", ticker=t.get("ticker", ""), type="avg"),
                price_daily_after_1d=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="1d", ticker=t.get("ticker", ""), type="day"),
                price_daily_after_7d=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="7d", ticker=t.get("ticker", ""), type="day"),
                price_daily_after_15d=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="15d", ticker=t.get("ticker", ""), type="day"),
                price_daily_after_1m=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="1m", ticker=t.get("ticker", ""), type="day"),
                price_daily_after_6m=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="6m", ticker=t.get("ticker", ""), type="day"),
                price_daily_after_1y=stocks_api_service(t.get("asset_description", ""), datetime.strptime(senator.get("date_recieved", ""), '%m/%d/%Y').date(), period="1y", ticker=t.get("ticker", ""), type="day"),
                
            )
            for senator in data or []
            for t in senator.get("transactions", [])
        ]
    ]
    return transactions if transactions else None

def main():
    transactions = get_transactions()
    transactions = normalize_data(transactions)
    transactions_json = {"transactions": transactions}

    print("Writing Transactions...")
    with open("transactions.json", "w") as f:
        dump(transactions_json, f, cls=TransactionEncoder)

if __name__ == "__main__":
    main()

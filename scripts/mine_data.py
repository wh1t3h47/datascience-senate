from json import dump
from src.models.TransactionModel import TransactionEncoder
from src.services.data_mining.data_mining_service import DataMiningService


def main():
    transactions = DataMiningService.get_transactions()
    transactions = DataMiningService.normalize_data(transactions)
    transactions_json = {"transactions": transactions}

    print("Writing Transactions...")
    with open("transactions.json", "w") as f:
        dump(transactions_json, f, cls=TransactionEncoder)

if __name__ == "__main__":
    main()

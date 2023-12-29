from src.services.machine_learning.machine_learning_service import machine_learning_service


def main() -> None:
    file_path: str = "transactions.json"
    machine_learning_service(file_path)


if __name__ == "__main__":
    main()

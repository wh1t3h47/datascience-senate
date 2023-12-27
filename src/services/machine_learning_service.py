from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from pandas import json_normalize, read_json, to_datetime
from sklearn.model_selection import cross_val_score
from os.path import abspath


def machine_learning_service(file_path):
    # Carregue os dados
    abs_path = abspath(file_path)
    df_denormalized = read_json(abs_path)
    transactions_list = df_denormalized['transactions'].tolist()

    # Normalize a lista de dicionários em um DataFrame
    df = json_normalize(transactions_list)
    #print(f'head => {df.head}')

    # Descarte dados com preços iguais a -1, -2 ou -3 (erros da API)
    price_columns = ["price_avarage_in_1d", "price_avarage_in_7d", "price_avarage_in_15d",
                     "price_avarage_in_1m", "price_avarage_in_6m", "price_avarage_in_1y"]
    df[price_columns] = df[price_columns].apply(lambda x: x.where(x >= 0))

    # Verifique se 'created_at' está presente e converta para formato apropriado
    if 'created_at' in df.columns:
        df['created_at'] = to_datetime(df['created_at'])

    # Verifique a existência das colunas de features
    feature_columns = ["price_avarage_in_1d", "price_avarage_in_7d", "price_avarage_in_15d",
                       "price_avarage_in_1m", "price_avarage_in_6m", "price_avarage_in_1y"]

    missing_columns = set(feature_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are missing in the DataFrame.")

    # Escolha as features relevantes
    features = df[feature_columns]

    # A coluna alvo é se o investimento foi repetido ou não (1 para repetido, 0 para não repetido)
    target = (df["type"] == "Repeat").astype(int)

    # Descarte linhas com valores nulos após a manipulação dos preços
    valid_indices = features[feature_columns].dropna().index
    features = features.loc[valid_indices]
    target = target.loc[valid_indices]

    # Divida os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Escolha o modelo (pode experimentar com outros modelos)
    model = DecisionTreeClassifier(random_state=42)

    # Treine o modelo
    model.fit(X_train, y_train)

    # Avalie o modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Avaliação usando validação cruzada
    cv_scores = cross_val_score(model, features, target, cv=5)
    avg_cv_accuracy = cv_scores.mean()

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "cross_val_accuracy": avg_cv_accuracy
    }

def main():
    file_path = "transactions.json"
    result = machine_learning_service(file_path)
    
    print("Resultado da Avaliação do Modelo:")
    print(f"  - Acurácia: {result['accuracy']:.2%}")
    print("\nRelatório de Classificação:")
    print(result['classification_report'])
    print(f"\nAcurácia da Validação Cruzada: {result['cross_val_accuracy']:.2%}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

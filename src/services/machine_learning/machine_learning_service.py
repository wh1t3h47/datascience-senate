from typing import List, Dict
from matplotlib.pyplot import plot, figure, xlabel, ylabel, title, legend, show
from pandas import json_normalize, read_json, DataFrame, Series, to_datetime
from os.path import abspath
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from src.models.TransactionModel import TransactionModel
from src.services.machine_learning.model_training_service import ModelTrainerService

def preprocess_data(df: DataFrame) -> DataFrame:
    transactions_list: List[TransactionModel] = df['transactions'].tolist()
    df_normalized: DataFrame = json_normalize(transactions_list)

    df_normalized = df_normalized.drop(columns=['asset_description', 'asset_type'])

    df_normalized['comment'] = df_normalized['comment'].replace('--', '')

    if 'created_at' in df_normalized.columns:
        df_normalized['created_at'] = to_datetime(df_normalized['created_at']).dt.strftime('%m-%d')

    return df_normalized


def feature_engineering(df: DataFrame) -> DataFrame:
    df['gain_avarage_in_1d'] = df['price_avarage_in_1d'].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)
    df['gain_daily_after_1d'] = df['price_daily_after_1d'].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)

    df['total_price_avarage'] = df[['price_avarage_in_1d', 'price_avarage_in_7d', 'price_avarage_in_15d']].sum(axis=1)
    df['total_price_daily_after'] = df[['price_daily_after_1d', 'price_daily_after_7d', 'price_daily_after_15d']].sum(axis=1)

    df['price_daily_after_1y'] = df['price_daily_after_1y'].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)

    df['toco_de_decisao'] = df['price_daily_after_15d'].apply(lambda x: x if x > 0 else 0)

    prices_columns: List[str] = ['price_avarage_in_1d', 'price_avarage_in_7d', 'price_avarage_in_15d',
                                 'price_avarage_in_1m', 'price_avarage_in_6m', 'price_avarage_in_1y',
                                 'price_daily_after_1d', 'price_daily_after_7d', 'price_daily_after_15d',
                                 'price_daily_after_1m', 'price_daily_after_6m', 'price_daily_after_1y']

    for col in prices_columns:
        df[f'{col}_variance'] = df[col].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)

    return df

def machine_learning_service(file_path: str) -> None:
    abs_path = abspath(file_path)
    df_denormalized = read_json(abs_path)

    df = preprocess_data(df_denormalized)
    df = feature_engineering(df)

    feature_columns: List[str] = ["gain_avarage_in_1d", "gain_daily_after_1d", "total_price_avarage", "total_price_daily_after",
                                  "price_daily_after_1y", "toco_de_decisao"]

    missing_columns: List[str] = list(set(feature_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are missing in the DataFrame.")

    features = df[feature_columns]
    target = (df["type"] == "Sale (Full)").astype(int)

    valid_indices = features[feature_columns].dropna().index
    features = features.loc[valid_indices]
    target = target.loc[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    param_dist_knn: Dict[str, List] = {
        'n_neighbors': [5, 10, 15, 20, 25],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_model = ModelTrainerService(KNeighborsClassifier(), "KNN", param_dist_knn)
    knn_model.train(X_train, y_train, n_iter=20)

    param_dist_rf: Dict[str, List] = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    param_dist_dt: Dict[str, List] = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    param_dist_nb: Dict[str, List] = {}  # GaussianNB não tem muitos hiperparâmetros

    rf_model = ModelTrainerService(RandomForestClassifier(random_state=42), "Random Forest", param_dist_rf)
    dt_model = ModelTrainerService(DecisionTreeClassifier(random_state=42), "Decision Tree", param_dist_dt)
    nb_model = ModelTrainerService(GaussianNB(), "GaussianNB", param_dist_nb)

    rf_model.train(X_train, y_train)
    dt_model.train(X_train, y_train)
    nb_model.train(X_train, y_train)

    # Combine all ROC curves in one plot
    figure()
    plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    knn_model.evaluate(X_test, y_test)
    rf_model.evaluate(X_test, y_test)
    dt_model.evaluate(X_test, y_test)
    nb_model.evaluate(X_test, y_test)

    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    title('Receiver Operating Characteristic Curve')
    legend()
    show()


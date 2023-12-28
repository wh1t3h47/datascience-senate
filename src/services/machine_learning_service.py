from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score
from sklearn.model_selection import train_test_split
from joblib import parallel_backend
from os.path import abspath
import numpy as np
import pandas as pd
from pandas import to_datetime

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    transactions_list = df['transactions'].tolist()
    df_normalized = pd.json_normalize(transactions_list)

    df_normalized = df_normalized.drop(columns=['asset_description', 'asset_type'])

    df_normalized['comment'] = df_normalized['comment'].replace('--', '')

    if 'created_at' in df_normalized.columns:
        df_normalized['created_at'] = to_datetime(df_normalized['created_at']).dt.strftime('%m-%d')

    return df_normalized

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['gain_avarage_in_1d'] = df['price_avarage_in_1d'].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)
    df['gain_daily_after_1d'] = df['price_daily_after_1d'].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)

    df['total_price_avarage'] = df[['price_avarage_in_1d', 'price_avarage_in_7d', 'price_avarage_in_15d']].sum(axis=1)
    df['total_price_daily_after'] = df[['price_daily_after_1d', 'price_daily_after_7d', 'price_daily_after_15d']].sum(axis=1)

    df['price_daily_after_1y'] = df['price_daily_after_1y'].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)

    df['toco_de_decisao'] = df['price_daily_after_15d'].apply(lambda x: x if x > 0 else 0)

    # Adicionando colunas de variância para todos os preços
    prices_columns = ['price_avarage_in_1d', 'price_avarage_in_7d', 'price_avarage_in_15d',
                      'price_avarage_in_1m', 'price_avarage_in_6m', 'price_avarage_in_1y',
                      'price_daily_after_1d', 'price_daily_after_7d', 'price_daily_after_15d',
                      'price_daily_after_1m', 'price_daily_after_6m', 'price_daily_after_1y']

    for col in prices_columns:
        df[f'{col}_variance'] = df[col].pct_change().apply(lambda x: 0 if abs(x) < 0.02 else x)

    return df

class ModelTrainer:
    def __init__(self, model, name, param_dist):
        self.model = model
        self.name = name
        self.param_dist = param_dist

    def train(self, X_train, y_train, n_iter=50):
        with parallel_backend('loky', n_jobs=10):
            random_search = RandomizedSearchCV(
                self.model,
                param_distributions=self.param_dist,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy',
                random_state=42
            )
            random_search.fit(X_train, y_train)

            best_params = random_search.best_params_
            self.model.set_params(**best_params)
            print(f"Best parameters for {self.name} after {n_iter} iterations: {best_params}")

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.4).astype(int)
        else:
            y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)
        precision = precision_score(y_test, y_pred, zero_division=1, average='weighted')
        f1 = f1_score(y_test, y_pred, zero_division=1, average='weighted')

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=cv)
        avg_cv_accuracy = cv_scores.mean()

        print(f"\nModel: {self.name}")
        print(f"  - Accuracy: {accuracy}")
        print(f"  - Precision: {precision}")
        print(f"  - F1 Score: {f1}")
        print(f"  - Classification Report:\n{report}")
        print(f"  - Cross-Validation Accuracy: {round(avg_cv_accuracy * 100):.0f}%")

def machine_learning_service(file_path: str) -> None:
    abs_path = abspath(file_path)
    df_denormalized = pd.read_json(abs_path)

    df = preprocess_data(df_denormalized)
    df = feature_engineering(df)

    feature_columns = ["gain_avarage_in_1d", "gain_daily_after_1d", "total_price_avarage", "total_price_daily_after", "price_daily_after_1y", "toco_de_decisao"]

    missing_columns = set(feature_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are missing in the DataFrame.")

    features = df[feature_columns]
    target = (df["type"] == "Sale (Full)").astype(int)

    valid_indices = features[feature_columns].dropna().index
    features = features.loc[valid_indices]
    target = target.loc[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Defina espaços de parâmetros adequados para o KNN
    param_dist_knn = {
        'n_neighbors': [5, 10, 15, 20, 25],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_model = ModelTrainer(KNeighborsClassifier(), "KNN", param_dist_knn)
    knn_model.train(X_train, y_train, n_iter=20)
    knn_model.evaluate(X_test, y_test)

    # Agora, os outros testes podem ser adicionados aqui
    # Defina espaços de parâmetros adequados para os outros modelos
    param_dist_rf = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    param_dist_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    param_dist_nb = {}  # GaussianNB não tem muitos hiperparâmetros

    rf_model = ModelTrainer(RandomForestClassifier(random_state=42), "Random Forest", param_dist_rf)
    dt_model = ModelTrainer(DecisionTreeClassifier(random_state=42), "Decision Tree", param_dist_dt)
    nb_model = ModelTrainer(GaussianNB(), "GaussianNB", param_dist_nb)

    rf_model.train(X_train, y_train)
    rf_model.evaluate(X_test, y_test)

    dt_model.train(X_train, y_train)
    dt_model.evaluate(X_test, y_test)

    nb_model.train(X_train, y_train)
    nb_model.evaluate(X_test, y_test)

def main() -> None:
    file_path = "transactions.json"
    machine_learning_service(file_path)

if __name__ == "__main__":
    main()

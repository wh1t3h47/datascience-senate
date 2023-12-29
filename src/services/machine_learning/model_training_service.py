from dataclasses import dataclass
from typing import Dict, List
from joblib import parallel_backend
from matplotlib.pyplot import plot
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, classification_report, f1_score, precision_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score


@dataclass
class ModelTrainerService:
    model: RandomForestClassifier
    name: str
    param_dist: Dict[str, List]

    def train(self, X_train: DataFrame, y_train: Series, n_iter: int = 1) -> None:
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

    def evaluate(self, X_test: DataFrame, y_test: Series) -> None:
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

        # Plot ROC curve
        if hasattr(self.model, 'predict_proba'):
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plot(fpr, tpr, label=f'{self.name} (AUC = {roc_auc:.2f})')


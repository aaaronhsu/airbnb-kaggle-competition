import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
import lightgbm as lgb

class OrdinalClassifier:
    def __init__(self, base_model='xgboost', model_params=None):
        self.base_model = base_model
        self.model_params = model_params or {}
        self.model = self._get_base_model()

    def _get_base_model(self):
        if self.base_model == 'xgboost':
            params = {
                'objective': 'multi:softmax',
                'num_class': 6,
                'min_child_weight': 1,
                'colsample_bytree': 0.6,
                'reg_lambda': 0.001,
                'eval_metric': ['mlogloss', 'merror'],
                'tree_method': 'hist',
                'random_state': 42,

                'colsample_bytree': 0.6,
                'learning_rate': 0.0156757227368663,
                'max_depth': 9,
                'min_child_weight': 1,
                'n_estimators': 882,
                'reg_alpha': 0.053626850522901205,
                'reg_lambda': 0.001,
                'subsample': 0.8744536442470929
            }
            params.update(self.model_params)
            return xgb.XGBClassifier(**params)

        elif self.base_model == 'lightgbm':
            params = {
                'objective': 'multiclass',
                'num_class': 6,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'metric': 'multi_logloss',
                'random_state': 42,
                'reg_alpha': 0.001,
                'reg_lambda': 0.001,

                'colsample_bytree': 0.7692725338291122,
                'learning_rate': 0.07485528354670547,
                'max_depth': 10,
                'min_child_samples': 2,
                'n_estimators': 100,
                'num_leaves': 87,
                'reg_alpha': 0.001,
                'reg_lambda': 0.001,
                'subsample': 0.8553076706085649
            }
            params.update(self.model_params)
            return lgb.LGBMClassifier(**params)



        elif self.base_model == 'random_forest':
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            params.update(self.model_params)
            return RandomForestClassifier(**params)

        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

    def fit(self, X, y, eval_set=None):
        """
        Fit the model to the training data.
        """
        if eval_set is not None:
            if self.base_model == 'xgboost':
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=True
                )
            elif self.base_model == 'lightgbm':
                self.model.fit(
                    X, y,
                    eval_set=eval_set
                )
            else:
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)
        return self




    def predict(self, X):
        """
        Make predictions on new data.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """
        Evaluate the model using metrics suitable for ordinal classification.
        """
        y_pred = self.predict(X)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Basic classification metrics
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)

        # Calculate adjacent category accuracy
        adjacent_correct = sum(abs(y - y_pred) <= 1)
        adjacent_accuracy = adjacent_correct / len(y)

        # Calculate per-class accuracy
        class_report = classification_report(y, y_pred, output_dict=True)

        return {
            'rmse': rmse,
            'accuracy': accuracy,
            'adjacent_accuracy': adjacent_accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

class EnsembleOrdinalClassifier:
    def __init__(self, models=None):
        if models is None:
            self.models = [
                get_model('xgboost'),
                get_model('lightgbm'),
                get_model('random_forest')
            ]
        else:
            self.models = models

    def fit(self, X, y, eval_set=None):
        """
        Fit all models in the ensemble.
        """
        for model in self.models:
            model.fit(X, y, eval_set=eval_set)
        return self

    def predict(self, X):
        """
        Make predictions using weighted voting.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        # Take the rounded average (since classes are ordinal)
        return np.round(np.mean(predictions, axis=0)).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities using averaged probabilities.
        """
        probas = [model.predict_proba(X) for model in self.models]
        return np.mean(probas, axis=0)

    def evaluate(self, X, y):
        """
        Evaluate the ensemble model.
        """
        y_pred = self.predict(X)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate adjacent category accuracy
        adjacent_correct = sum(abs(y - y_pred) <= 1)
        adjacent_accuracy = adjacent_correct / len(y)

        return {
            'rmse': rmse,
            'accuracy': accuracy_score(y, y_pred),
            'adjacent_accuracy': adjacent_accuracy,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }

def get_model(model_type='xgboost', params=None):
    """
    Factory function to create a model instance.
    """
    return OrdinalClassifier(base_model=model_type, model_params=params)

def get_ensemble_model():
    """
    Factory function to create an ensemble model.
    """
    return EnsembleOrdinalClassifier()

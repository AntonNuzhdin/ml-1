from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from scipy.special import softmax


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        bootstrap_indexes = np.random.choice(x.shape[0], size=int(self.subsample * x.shape[0]))
        x_bootstrap = x[bootstrap_indexes]

        s_i = -self.loss_derivative(y, predictions)

        current_model = self.base_model_class(**self.base_model_params)
        current_model.fit(x_bootstrap, s_i[bootstrap_indexes])

        pred = current_model.predict(x)

        best_gamma = self.find_optimal_gamma(y, predictions, pred)

        self.gammas.append(best_gamma)
        self.models.append(current_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        steps_after_best_loss = 0
        best_valid_loss = np.inf

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.gammas[-1] * self.learning_rate * self.models[-1].predict(x_train)
            valid_predictions += self.gammas[-1] * self.learning_rate * self.models[-1].predict(x_valid)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if self.early_stopping_rounds is not None and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                steps_after_best_loss = 0
            else:
                steps_after_best_loss += 1

            if self.early_stopping_rounds is not None:
                if steps_after_best_loss >= self.early_stopping_rounds:
                    break

        if self.plot:
            plt.figure(figsize=(10, 7))
            plt.plot(np.arange(len(self.history['train_loss'])), self.history['train_loss'], label='Train loss')
            plt.plot(np.arange(len(self.history['valid_loss'])), self.history['valid_loss'], label='Valid loss')
            plt.xlabel('Number of estimators')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Train and validate losses')
            plt.show()

    def predict_proba(self, x):
        prediction = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            prediction += gamma * self.learning_rate * model.predict(x)
        prediction = self.sigmoid(prediction)
        invert_prediction = 1 - prediction
        return np.column_stack((invert_prediction, prediction))

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importances = np.zeros(self.models[0].feature_importances_.shape)
        for model in self.models:
            feature_importances += model.feature_importances_
        feature_importances /= len(self.models)
        feature_importances / np.sum(feature_importances)
        return feature_importances

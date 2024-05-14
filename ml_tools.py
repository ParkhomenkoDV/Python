import sys
import os
from tqdm import tqdm
import warnings
import pickle
import joblib

import multiprocessing as mp
import threading as th

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import (Normalizer,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, QuantileTransformer, PowerTransformer)

from sklearn.linear_model import (SGDClassifier, SGDOneClassSVM, RidgeClassifier, RidgeClassifierCV,
                                  PassiveAggressiveClassifier)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                                  OrthogonalMatchingPursuit,
                                  BayesianRidge, ARDRegression, SGDRegressor, RANSACRegressor, GammaRegressor,
                                  PoissonRegressor, HuberRegressor,
                                  TweedieRegressor, LogisticRegression, QuantileRegressor, TheilSenRegressor)
from sklearn.neighbors import (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
                               RadiusNeighborsClassifier, RadiusNeighborsRegressor)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree import plot_tree

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                              StackingRegressor, VotingRegressor)

from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error, max_error,
                             coverage_error,
                             mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_log_error, root_mean_squared_log_error)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score,
                             d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
                             adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
                             jaccard_score, consensus_score, v_measure_score, brier_score_loss, d2_tweedie_score,
                             cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
                             average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
                             top_k_accuracy_score, calinski_harabasz_score)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc

from tools import export2

SCALERS = (Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer)


class Model:
    """Абстрактный класс модели"""

    NEIGHBORS = [NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor,
                 RadiusNeighborsClassifier, RadiusNeighborsRegressor]
    TREE_CLASSIFIERS = [DecisionTreeClassifier, ExtraTreeClassifier]
    TREE_REGRESSORS = [DecisionTreeRegressor, ExtraTreeRegressor]
    ENSEMBLE_CLASSIFIERS = [RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
                            GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
                            StackingClassifier, VotingClassifier]
    ENSEMBLE_REGRESSORS = [RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                           GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                           StackingRegressor, VotingRegressor]

    ALL_MODELS = (NEIGHBORS +
                  TREE_CLASSIFIERS + TREE_REGRESSORS +
                  ENSEMBLE_CLASSIFIERS + ENSEMBLE_REGRESSORS)

    def __init__(self, model=None):

        if not model:
            self.__model = None

        elif type(model) is str:
            model = model.strip().replace('()', '')
            assert model in (class_model.__name__ for class_model in self.ALL_MODELS), \
                f'model in {[class_model.__name__ for class_model in self.ALL_MODELS]}'

            self.__model = next((class_model() for class_model in self.ALL_MODELS if model == class_model.__name__),
                                None)

        elif type(model) in self.ALL_MODELS:
            self.__model = model

        else:
            raise AssertionError(
                f'type(model) in {[str.__name__] + [class_model.__name__ for class_model in self.ALL_MODELS]}')

    def __call__(self):
        return self.__model

    def __repr__(self):
        return str(self.__model)

    @property
    def intercept_(self):
        try:
            return self.__model.intercept_
        except Exception as e:
            print(e)

    @property
    def coef_(self):
        try:
            return self.__model.coef_
        except Exception as e:
            print(e)

    @property
    def expression(self):
        try:
            return f'y = {self.intercept_} + {" + ".join([f"{num} * x{i + 1}" for i, num in enumerate(self.coef_)])}'
        except Exception as e:
            print(e)

    @property
    def feature_importances_(self):
        try:
            return self.__model.feature_importances_
        except Exception as e:
            print(e)

    def fit(self, x, y):
        self.__model.fit(x, y)

    def predict(self, x):
        return self.__model.predict(x)

    def prediction(self, y_true, y_possible, suptitle='Prediction', bins=40, savefig=False):

        fg = plt.figure(figsize=(12, 8))
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
        gs = fg.add_gridspec(1, 2)

        fg.add_subplot(gs[0, 0])
        plt.grid(True)
        plt.hist(y_true - y_possible, bins=bins)

        fg.add_subplot(gs[0, 1])
        plt.grid(True)
        plt.scatter(y_true, y_possible, color='red')
        plt.plot(y_true, y_true, color='blue')

        if savefig: export2(plt, file_name=suptitle, file_extension='png')

    def plot_tree(self, **kwargs):
        try:
            plot_tree(self.__model, filled=kwargs.get('filled', True))
        except Exception as e:
            if e: print(e)

    def fit_all(self, x, y, exceptions=True):

        '''
        def fit_model(Model):
            return Model().fit(x, y)

        with mp.Pool(mp.cpu_count) as pool:
            results = pool.map(fit_model, self.MODELS)

        return results
        '''

        warnings.filterwarnings('ignore')

        result = list()
        for class_model in tqdm(self.ALL_MODELS):
            try:
                model = class_model().fit(x, y)
                result.append(Model(model))
            except Exception as e:
                if exceptions: print(e)

        warnings.filterwarnings('default')

        return result

    def errors(self, y_true, y_predict, exceptions=True) -> dict[str:float]:
        errors = dict()
        for error in self.ALL_ERRORS:
            try:
                errors[error.__name__] = error(y_true, y_predict)
            except Exception as e:
                if exceptions: print(e)
        return errors

    def scores(self, y_true, y_predict, exceptions=True) -> dict[str:float]:
        scores = dict()
        for score in self.ALL_SCORES:
            try:
                scores[score.__name__] = score(y_true, y_predict)
            except Exception as e:
                if exceptions: print(e)
        return scores

    def save(self, path: str, method: str = 'pickle') -> None:
        """Сохранение модели"""
        assert type(method) is str, 'type(method) is str'
        method = method.strip().lower()
        assert method in ("pickle", "joblib"), 'method in ("pickle", "joblib")'

        if method == 'pickle':
            pickle.dump(self.__model, open(path, 'wb'))
        elif method == 'joblib':
            joblib.dump(self.__model, open(path, 'wb'))

    def load(self, path: str, method: str = 'pickle'):
        """Загрузка модели"""
        assert type(method) is str, 'type(method) is str'
        method = method.strip().lower()
        assert method in ("pickle", "joblib"), 'method in ("pickle", "joblib")'

        if method == 'pickle':
            self.__model = pickle.load(open(path, 'rb'))
        elif method == 'joblib':
            self.__model = joblib.load(open(path, 'rb'))

        return self


class Classifier(Model):
    """Модель классификатора"""
    LINEAR_MODEL_CLASSIFIERS = [SGDClassifier, SGDOneClassSVM, RidgeClassifier, RidgeClassifierCV,
                                PassiveAggressiveClassifier]
    TREE_CLASSIFIERS = [DecisionTreeClassifier, ExtraTreeClassifier]
    MODELS = ()

    ALL_SCORES = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score,
                  d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
                  adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
                  jaccard_score, consensus_score, v_measure_score, brier_score_loss, d2_tweedie_score,
                  cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
                  average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
                  top_k_accuracy_score, calinski_harabasz_score]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def confusion_matrix(self, y_true, y_predicted):
        """Матрица путаницы"""
        return confusion_matrix(y_true, y_predicted, labels=self.__model.classes_)

    def confusion_matrix_plot(self, y_true, y_predicted, title='confusion_matrix', **kwargs):
        """График матрицы путаницы"""
        cm = self.confusion_matrix(y_true, y_predicted)
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.__model.classes_)
        # plt.figure(figsize=kwargs.get('figsize', (12, 12)))
        # plt.title(title, fontsize=16, fontweight='bold')
        cmd.plot()
        plt.show()

    def precision_recall_curve(self, y_true, y_predicted, **kwargs):
        """График precision-recall"""
        precision, recall, threshold = precision_recall_curve(y_true, y_predicted)
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.title(kwargs.get('title', 'precision recall curve'), fontsize=16, fontweight='bold')
        plt.plot(precision, recall, color='blue', label='PR')
        plt.legend(loc='lower left')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.xlabel('precision', fontsize=14)
        plt.ylabel('recall', fontsize=14)
        if kwargs.get('savefig', False):
            export2(plt, file_name=kwargs.get('title', 'precision recall curve'), file_extension='png')
        plt.show()

    def roc_curve(self, y_true, y_predicted, **kwargs):
        """График ROC"""
        fpr, tpr, threshold = roc_curve(y_true, y_predicted)
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.title(kwargs.get('title', 'roc curve'), fontsize=16, fontweight='bold')
        plt.plot(fpr, tpr, color='blue', label=f'ROC-AUC = {auc(fpr, tpr)}')
        plt.plot([0, 1], [0, 1], color='red')
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.xlabel('fpr', fontsize=14)
        plt.ylabel('tpr', fontsize=14)
        if kwargs.get('savefig', False): export2(plt, file_name=kwargs.get('title', 'roc curve'), file_extension='png')
        plt.show()


class Regressor(Model):
    MODELS = ()

    LINEAR_MODEL_REGRESSORS = [LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                               OrthogonalMatchingPursuit,
                               BayesianRidge, ARDRegression, SGDRegressor, RANSACRegressor, GammaRegressor,
                               PoissonRegressor, HuberRegressor,
                               TweedieRegressor, LogisticRegression, QuantileRegressor, TheilSenRegressor]

    ALL_ERRORS = [mean_absolute_error, mean_squared_error, root_mean_squared_error, max_error,
                  coverage_error,
                  mean_absolute_percentage_error, median_absolute_error,
                  mean_squared_log_error, root_mean_squared_log_error]

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)


class Clusterizer(Model):
    MODELS = (DBSCAN,)

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)


def classifier_or_regressor(model) -> str:
    model_type = list()
    try:
        model_type.append(type(model).__name__.lower())
    except:
        pass
    try:
        model_type.append(model.__name__.lower())
    except:
        pass

    if 'cla' in model_type: return 'cla'
    if 'reg' in model_type: return 'reg'
    return ''


class Stacking:

    # TODO: доделать
    def __init__(self, models: tuple | list):
        self.__stacking = StackingClassifier() if 'cla' in type(model).__name__.lower() else StackingRegressor()

    def __call__(self):
        return self.__stacking

    def fit(self, x, y):
        self.__stacking.fit(x, y)


class Bagging:

    def __init__(self, model, **kwargs):
        bagging_type = classifier_or_regressor(model)
        if bagging_type == 'cla':
            self.__bagging = BaggingClassifier(model, **kwargs)
        elif bagging_type == 'reg':
            self.__bagging = BaggingRegressor(model, **kwargs)
        else:
            raise 'type(model) is "classifier" or "regressor"'

    def __call__(self):
        return self.__bagging

    def fit(self, x, y):
        self.__bagging.fit(x, y)

    def predict(self, x):
        return self.__bagging.predict(x)


class Boosting:

    # TODO: доделать
    def __init__(self):
        pass


if __name__ == '__main__':
    model = Model()

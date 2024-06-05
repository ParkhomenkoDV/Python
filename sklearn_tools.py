import sys
import os
from tqdm import tqdm
from colorama import Fore
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

from sklearn.model_selection import train_test_split, GridSearchCV

import decorators
from tools import export2

SCALERS = (Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer)


def gini(y_true, y_predicted) -> float:
    """Критерий Джинни"""
    return 2 * roc_auc_score(y_true, y_predicted) - 1


class Model:
    """Модель ML"""
    # линейные модели
    LINEAR_MODEL_CLASSIFIERS = [SGDClassifier, SGDOneClassSVM, RidgeClassifier, RidgeClassifierCV,
                                PassiveAggressiveClassifier]
    LINEAR_MODEL_REGRESSORS = [LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                               OrthogonalMatchingPursuit,
                               BayesianRidge, ARDRegression, SGDRegressor, RANSACRegressor, GammaRegressor,
                               PoissonRegressor, HuberRegressor,
                               TweedieRegressor, LogisticRegression, QuantileRegressor, TheilSenRegressor]

    # деревья
    TREE_CLASSIFIERS = [DecisionTreeClassifier, ExtraTreeClassifier]
    TREE_REGRESSORS = [DecisionTreeRegressor, ExtraTreeRegressor]

    # ансамбли
    ENSEMBLE_CLASSIFIERS = [RandomForestClassifier, ExtraTreesClassifier,
                            BaggingClassifier,
                            GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
                            StackingClassifier,
                            VotingClassifier]
    ENSEMBLE_REGRESSORS = [RandomForestRegressor, ExtraTreesRegressor,
                           BaggingRegressor,
                           GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                           StackingRegressor,
                           VotingRegressor]
    STACKINGS_CLASSIFIERS = [StackingClassifier]
    STACKINGS_REGRESSORS = [StackingRegressor]
    BAGGING_CLASSIFIERS = [BaggingClassifier]
    BAGGING_REGRESSORS = [BaggingRegressor]
    BOOSTING_CLASSIFIERS = [GradientBoostingClassifier]
    BOOSTING_REGRESSORS = [GradientBoostingRegressor]
    # соседи
    NEIGHBORS_CLASSIFIERS = [KNeighborsClassifier, RadiusNeighborsClassifier]
    NEIGHBORS_REGRESSORS = [KNeighborsRegressor, RadiusNeighborsRegressor]

    # стэкинги
    STACKINGS = STACKINGS_CLASSIFIERS + STACKINGS_REGRESSORS
    # бэггинги
    BAGGINGS = BAGGING_CLASSIFIERS + BAGGING_REGRESSORS
    # бустинги
    BOOSTINGS = BOOSTING_CLASSIFIERS + BOOSTING_REGRESSORS

    # классификаторы
    CLASSIFIERS = LINEAR_MODEL_CLASSIFIERS + TREE_CLASSIFIERS + ENSEMBLE_CLASSIFIERS
    # регрессоры
    REGRESSORS = LINEAR_MODEL_REGRESSORS + TREE_REGRESSORS + ENSEMBLE_REGRESSORS
    # кластеризаторы
    CLUSTERIZERS = NEIGHBORS_CLASSIFIERS + NEIGHBORS_REGRESSORS + [NearestNeighbors] + [DBSCAN]

    # все модели ML
    MODELS = set(CLASSIFIERS + REGRESSORS + CLUSTERIZERS)

    SCORES = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score,
              d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
              adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
              jaccard_score, consensus_score, v_measure_score, brier_score_loss, d2_tweedie_score,
              cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
              average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
              top_k_accuracy_score, calinski_harabasz_score]

    ERRORS = [mean_absolute_error, mean_squared_error, root_mean_squared_error, max_error,
              coverage_error,
              mean_absolute_percentage_error, median_absolute_error,
              mean_squared_log_error, root_mean_squared_log_error]

    def __init__(self, model=None):
        """Инициализация модели ML None/названием модели/объектом ML"""
        if model is None:
            self.__model = None
        elif type(model) in self.MODELS:
            self.__model = model
        else:
            raise AssertionError(
                f'type(model) in {[str.__name__] + [class_model.__name__ for class_model in self.MODELS]}')

    def __call__(self):
        return self.__model

    def __str__(self):
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
        """Обучение модели"""
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

    @decorators.warns('ignore')
    def fit_all(self, x, y, exceptions=True):
        """Обучение всех моделей"""

        '''
        def fit_model(Model):
            return Model().fit(x, y)

        with mp.Pool(mp.cpu_count) as pool:
            results = pool.map(fit_model, self.MODELS)

        return results
        '''

        result = list()
        for class_model in tqdm(self.MODELS):
            try:
                model = class_model().fit(x, y)
                result.append(Model(model))
            except Exception as e:
                if exceptions: print(e)

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

    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)


class Clusterizer(Model):
    """Кластеризатор"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

    def show_elbow(self):
        """Метод локтя"""

        plt.xlabel('n_clusters')
        plt.ylabel('score')
        plt.show()


# TODO: добавить в класс Model как статический метод
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
        self.__stacking = StackingClassifier() if 'cla' in type(models).__name__.lower() else StackingRegressor()

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


class Boosting(Classifier, Regressor):

    def __init__(self, model, *args, **kwargs):
        assert type(model) in self.BOOSTINGS
        if 'classifier' in model.__class__.__name__.lower():
            super(Boosting, self).__init__(model)
        elif 'regressor' in model.__class__.__name__.lower():
            super(Boosting, self).__init__(model)
        else:
            raise


if __name__ == '__main__':
    if 0:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        classifier = Classifier()
        classifier.fit_all(data.data, data.target)
    if 0:
        from sklearn.datasets import load_wine

        data = load_wine(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        regressor = Regressor()
        regressor.fit_all(data.data, data.target)
    if 0:
        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        clusterizer = Clusterizer()
        clusterizer.fit_all(data.data, data.target)
    if 0:
        stacking = Stacking()
    if 0:
        bagging = Bagging()
    if 0:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        print(pd.concat([data.data, data.target], axis=1))

        x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

        boosting = Boosting(GradientBoostingClassifier())
        boosting.fit(x_train, y_train)

        y_pred = boosting.predict(x_test)

        print(f'gini: {gini(y_test, y_pred)}')

    if 1:
        print(Fore.YELLOW + GridSearchCV.__name__ + Fore.RESET)

        from sklearn.datasets import load_breast_cancer
        from sklearn.datasets import load_wine
        from sklearn.datasets import load_iris

        datas = [load_breast_cancer(as_frame=True), load_wine(as_frame=True), load_iris(as_frame=True)]
        models = [GradientBoostingClassifier(), GradientBoostingRegressor(), DBSCAN()]
        param_grids = [
            {'n_estimators': list(range(25, 150 + 1, 25)),
             'max_depth': list(range(3, 15, 3)),
             'random_state': [0]},
            {'n_estimators': list(range(25, 150 + 1, 25)),
             'max_depth': list(range(3, 15, 3)),
             'random_state': [0]},
            {'n_estimators': list(range(25, 150 + 1, 25)),
             'max_depth': list(range(3, 15, 3)),
             'random_state': [0]}
        ]

        for i in range(3):
            x_train, x_test, y_train, y_test = train_test_split(datas[i].data, datas[i].target, test_size=0.25)

            gscv = GridSearchCV(models[i], param_grid=param_grids[i], n_jobs=-1)
            gscv.fit(x_train, y_train)

            print(f'best params: {gscv.best_params_}')
            best_model = gscv.best_estimator_
            print(f'best estimator: {best_model}')
            try:
                print(f'train score: {best_model.score(x_train, y_train)}')
                print(f'test score: {best_model.score(x_test, y_test)}')
            except:
                pass

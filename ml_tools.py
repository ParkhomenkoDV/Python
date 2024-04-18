import sys
import os
from tqdm import tqdm
import pickle
import warnings

import multiprocessing as mp
import threading as th

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder, OrdinalEncoder

from sklearn.preprocessing import (Normalizer,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, QuantileTransformer, PowerTransformer)

from sklearn.model_selection import train_test_split

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

SCALERS = (Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer)

from tools import export2


def get_one_hot(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
    return df


def get_label(df, cols):
    df = df.copy()
    for col in cols:
        le = LabelEncoder()
        labels = le.fit_transform(df[col])
        df[col] = labels
    return df


def get_count(df, cols):
    df = df.copy()

    for col in cols:
        df[col] = df[col].astype('str')

    ce = CountEncoder(handle_unknown=-1)
    ce.fit(df[cols])
    df[cols] = ce.transform(df[cols])
    return df


def img_show(img, title='image', figsize=(12, 12)):
    plt.figure(figsize=figsize)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(title)
    plt.axis("off")
    plt.show()


def training_plot(history, figsize=(12, 9), savefig=False):
    num_metrics = len(history.history.keys()) // 2

    fg = plt.figure(figsize=figsize)  # размер в дюймах
    gs = fg.add_gridspec(1, num_metrics)  # строки, столбцы
    fg.suptitle('Training and Validation', fontsize=16, fontweight='bold')

    for i in range(num_metrics):
        metric_name = list(history.history.keys())[i]
        val_metric_name = list(history.history.keys())[i + num_metrics]

        fg.add_subplot(gs[0, i])  # позиция графика
        plt.grid(True)  # сетка
        plt.plot(history.history[metric_name], color='blue', label=metric_name)
        plt.plot(history.history[val_metric_name], color='red', label=val_metric_name)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Error', fontsize=12)
        plt.xlim(0, max(history.epoch))
        plt.ylim(0, 2 * np.mean([history.history[metric_name][-1], history.history[val_metric_name][-1]]))
        plt.legend()
    if savefig: export2(plt, file_name='training_plot', file_extension='png')
    plt.show()


def predictions_plot(y_true, y_predict, figsize=(12, 9), bins=40, savefig=False):
    fg = plt.figure(figsize=figsize)
    gs = fg.add_gridspec(1, 2)
    fg.suptitle('Predictions', fontsize=16, fontweight='bold')

    fg.add_subplot(gs[0, 0])
    plt.grid(True)
    plt.hist(y_predict - y_true, bins=bins)
    plt.xlabel('Predictions Error', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    fg.add_subplot(gs[0, 1])
    plt.grid(True)
    plt.scatter(y_true, y_predict, color='blue')
    lims = (min(*y_true, *y_predict), max(*y_true, *y_predict))
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims, color='red')
    plt.xlabel('True values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    if savefig: export2(plt, file_name='predictions_plot', file_extension='png')
    plt.show()


def pairplot(df, figsize=(9, 9), savefig=False):
    sns.set(style='whitegrid')
    g = sns.PairGrid(df, diag_sharey=False, height=4)
    g.fig.set_size_inches(figsize)
    g.map_diag(sns.kdeplot, lw=2)
    g.map_lower(sns.scatterplot, s=25, edgecolor="k", linewidth=0.5, alpha=0.4)
    g.map_lower(sns.kdeplot, cmap='plasma', n_levels=6, alpha=0.5)
    plt.tight_layout()
    if savefig: export2(plt, file_name='pair_plot', file_extension='png')


class DataFrame(pd.DataFrame):

    def detect_outliers(self, method: str = '3sigma'):
        assert_sms = 'Incorrect assert '
        assert type(self) is DataFrame, assert_sms + 'type(df) is DataFrame'
        assert type(method) is str, assert_sms + 'type(method) is str'
        method = method.strip().lower()
        assert method in ('3sigma', 'tukey'), assert_sms + 'method in ("3sigma", "Tukey")!'

        outliers = pd.DataFrame()
        for col in self.select_dtypes(include='number').columns:
            if method == '3sigma':
                mean = self[col].mean()
                std = self[col].std()
                lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
            elif method == 'tukey':
                q1, q3 = self[col].quantile(0.25), self[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            col_outliers = self[(self[col] < lower_bound) | (self[col] > upper_bound)]
            outliers = pd.concat([outliers, col_outliers])
        return outliers

    def find_drop_features(self, threshold: float = 0.9) -> list:
        com_m = self.corr().abs()

        # верхний треугольник матрицы корреляции без диагонали
        uptriangle = com_m.where(np.triu(np.ones(com_m.shape), k=1).astype(bool))

        # индексы столбцов с корреляцией выше порога
        to_drop = [column for column in uptriangle.columns[1:]
                   if any(threshold <= uptriangle[column]) or all(uptriangle[column].isna())]

        return to_drop


class Model:
    LINEAR_MODEL_CLASSIFIERS = [SGDClassifier, SGDOneClassSVM, RidgeClassifier, RidgeClassifierCV,
                                PassiveAggressiveClassifier]
    LINEAR_MODEL_REGRESSORS = [LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
                               OrthogonalMatchingPursuit,
                               BayesianRidge, ARDRegression, SGDRegressor, RANSACRegressor, GammaRegressor,
                               PoissonRegressor, HuberRegressor,
                               TweedieRegressor, LogisticRegression, QuantileRegressor, TheilSenRegressor]
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

    ALL_MODELS = (LINEAR_MODEL_CLASSIFIERS + LINEAR_MODEL_REGRESSORS +
                  NEIGHBORS +
                  TREE_CLASSIFIERS + TREE_REGRESSORS +
                  ENSEMBLE_CLASSIFIERS + ENSEMBLE_REGRESSORS)

    ALL_ERRORS = [mean_absolute_error, mean_squared_error, root_mean_squared_error, max_error,
                  coverage_error,
                  mean_absolute_percentage_error, median_absolute_error,
                  mean_squared_log_error, root_mean_squared_log_error]

    ALL_SCORES = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score,
                  d2_absolute_error_score, ndcg_score, rand_score, dcg_score, fbeta_score,
                  adjusted_rand_score, silhouette_score, completeness_score, homogeneity_score,
                  jaccard_score, consensus_score, v_measure_score, brier_score_loss, d2_tweedie_score,
                  cohen_kappa_score, d2_pinball_score, mutual_info_score, adjusted_mutual_info_score,
                  average_precision_score, label_ranking_average_precision_score, balanced_accuracy_score,
                  top_k_accuracy_score, calinski_harabasz_score]

    def __init__(self, model=None):

        assert_sms = 'Incorrect assert:'

        if not model:
            self.__model = None

        elif type(model) is str:
            model = model.strip().replace('()', '')
            assert model in (class_model.__name__ for class_model in self.ALL_MODELS), \
                f'{assert_sms} model in {[class_model.__name__ for class_model in self.ALL_MODELS]}'

            self.__model = next((class_model() for class_model in self.ALL_MODELS if model == class_model.__name__),
                                None)

        elif type(model) in self.ALL_MODELS:
            self.__model = model

        else:
            raise AssertionError(
                f'{assert_sms} type(model) in {[str.__name__] + [class_model.__name__ for class_model in self.ALL_MODELS]}')

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
        plt.scatter(y_true, y_possible, c='red')
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

    def save(self, path: str) -> None:
        pickle.dump(self.__model, open(path, 'wb'))

    def load(self, path: str):
        self.__model = pickle.load(open(path, 'rb'))
        return self


if __name__ == '__main__':
    df = DataFrame(pd.read_csv('airfoil_self_noise.dat', sep="\t", header=None))
    print(df)
    print(df.detect_outliers())
    print(df.find_drop_features())

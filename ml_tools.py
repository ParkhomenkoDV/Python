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

from sklearn.inspection import permutation_importance

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


class DataFrame(pd.DataFrame):
    """Расширенный класс pandas.DataFrame"""

    def encode_one_hot(self, columns: list[str], inplace=False):
        for column in columns:
            dummies = pd.get_dummies(self[column], prefix=column)
            self = pd.concat([self, dummies], axis=1, inplace=True)  # FIXME: self переопределен!
        return self

    def detect_outliers(self, method: str = '3sigma'):
        assert_sms = 'Incorrect assert'
        assert type(self) is DataFrame, f'{assert_sms} type(df) is DataFrame'
        assert type(method) is str, f'{assert_sms} type(method) is str'
        method = method.strip().lower()
        assert method in ('3sigma', 'tukey'), f'{assert_sms} method in ("3sigma", "Tukey")!'

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
        return DataFrame(outliers)

    def find_corr_features(self, threshold: float = 0.85) -> list[str]:
        com_m = self.corr().abs()

        # верхний треугольник матрицы корреляции без диагонали
        uptriangle = com_m.where(np.triu(np.ones(com_m.shape), k=1).astype(bool))

        # индексы столбцов с корреляцией выше порога
        to_drop = [column for column in uptriangle.columns[1:]
                   if any(threshold <= uptriangle[column]) or all(uptriangle[column].isna())]

        return to_drop

    def L1_importance(self):
        scaler, model = StandardScaler(), Lasso()

        '''X_sc = StandardScaler().fit_transform(X)  # преобразование данных
        lg_l, pred_l = [], []

        list_l = list(2 ** np.linspace(-10, 10, 100))

        # строим n-ое кол-во моделей Лассо, меняя коэффициент регуляризации, сохраняя модель и коэффициенты
        for i in range(len(list_l)):
            m_l = Lasso(alpha=list_l[i]).fit(X_sc, Y)
            lg_l.append(m_l)
            pred_l.append(m_l.coef_)

        # рисуем отмасштабированные признаки на одном графике
        plt.figure(figsize=(12, 9))
        x_l = np.linspace(0, len(pred_l), len(pred_l))
        for i in np.vstack(pred_l).T:
            plt.plot(x_l, np.sign(i) * np.abs(i))

        plt.ylim(-0.05, 0.2)
        plt.legend(names)
        plt.grid()'''

    def L1_importance_plot(self):
        pass

    def mutual_info_score(self, target: str) -> dict[str:float]:
        """Взаимная информация"""
        result = dict()
        for column in self.drop([target], axis=1):
            result[column] = mutual_info_score(self[column], self[target])
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return dict(result)

    def permutation_importance(self, target: str):
        """Перемешивающий подход"""
        model = RandomForestClassifier()
        try:
            result = permutation_importance(model, self.drop([target], axis=1), self[target])
            return pd.Series(result['importances_mean'], index=self.columns[:-1]).sort_values(ascending=True)
        except Exception as e:
            print(e)

    def feature_importances_(self, target: str):
        """Важные признаки для классификации"""
        model = RandomForestClassifier()
        try:
            model.fit(self.drop([target], axis=1), self[target])
            return pd.Series(model.features_importances_, index=self.columns[:-1]).sort_values(ascending=True)
        except Exception as e:
            print(e)

    def corrplot(self, figsize=(12, 12), title='Correlation', fmt=3, savefig=False):
        """Тепловая карта матрицы корреляции"""
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=16, fontweight='bold')
        sns.heatmap(self.corr(), annot=True, fmt=f'.{fmt}f')
        if savefig: export2(plt, file_name=title, file_extension='png')

    def pairplot(self, figsize=(9, 9), savefig=False):
        sns.set(style='whitegrid')
        g = sns.PairGrid(self, diag_sharey=False, height=4)
        g.fig.set_size_inches(figsize)
        g.map_diag(sns.kdeplot, lw=2)
        g.map_lower(sns.scatterplot, s=25, edgecolor="k", linewidth=0.5, alpha=0.4)
        g.map_lower(sns.kdeplot, cmap='plasma', n_levels=6, alpha=0.5)
        plt.tight_layout()
        if savefig: export2(plt, file_name='pair_plot', file_extension='png')

    def histplot(self, figsize=(9, 9), bins=40, savefig=False):
        self.hist(figsize=figsize, bins=bins)
        if savefig: export2(plt, file_name='histplot', file_extension='png')

    def boxplot(self, figsize=(12, 9), title='boxplot', scale=False, fill=True, grid=True, savefig=False):
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(grid)
        if not scale:
            sns.boxplot(self, fill=fill)
        else:
            sns.boxplot(pd.DataFrame(StandardScaler().fit_transform(self), columns=self.columns), fill=fill)
        if savefig: export2(plt, file_name='boxplot', file_extension='png')

    def train_test_split(self, test_size, shuffle=True, random_state=0):
        return train_test_split(self, test_size=test_size, shuffle=shuffle, random_state=random_state)


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
    print(df.find_corr_features())

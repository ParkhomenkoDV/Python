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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder

from sklearn.preprocessing import (Normalizer,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, QuantileTransformer, PowerTransformer)

from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit

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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from decorators import ignore_warnings
from tools import isiter, export2

SCALERS = (Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer)


def img_show(img, title='image', figsize=(12, 12)):
    plt.figure(figsize=figsize)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(title)
    plt.axis("off")
    plt.show()


class DataFrame(pd.DataFrame):
    """Расширенный класс pandas.DataFrame"""

    def __init__(self, *args, **kwargs):
        super(DataFrame, self).__init__(*args, **kwargs)

    def encode_label(self, columns: list[str], drop=False, inplace=False):
        """Преобразование n категорий в числа от 1 до n"""
        df = DataFrame()
        for column in columns:
            le = LabelEncoder()
            labels = le.fit_transform(self[column])
            df[column + '_label'] = labels
        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def encode_one_hot(self, columns: list[str], drop=False, inplace=False):
        """Преобразование n значений каждой категории в n бинарных категорий"""
        ohe = OneHotEncoder(handle_unknown='ignore')
        dummies = ohe.fit_transform(self[columns])
        df = DataFrame(dummies.toarray(), columns=ohe.get_feature_names_out())
        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def encode_count(self, columns: list[str], drop=False, inplace=False):
        """Преобразование значений каждой категории в количество этих значений"""
        df = DataFrame()
        for column in columns:
            column_count = self[column].value_counts().to_dict()
            df[column + '_count'] = self[column].map(column_count)
        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def encode_ordinal(self, columns: list[str], drop=False, inplace=False):
        """Преобразование категориальных признаков в числовые признаки с учетом порядка или их весов"""
        df = DataFrame()
        for column in columns:
            oe = OrdinalEncoder()
            df[column + '_ordinal'] = DataFrame(oe.fit_transform(self[[column]]))
        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def encode_target(self, columns: list[str], drop=False, inplace=False):
        """"""
        df = DataFrame()
        for column in columns:
            te = TargetEncoder()
            df[column + '_target'] = DataFrame(te.fit_transform(X=df.nom_0, y=df.Target))
        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def detect_outliers(self, method: str = '3sigma'):
        assert_sms = 'Incorrect assert'
        assert type(self) is DataFrame, f'{assert_sms} type(df) is DataFrame'
        assert type(method) is str, f'{assert_sms} type(method) is str'
        method = method.strip().lower()
        assert method in ('3sigma', 'tukey'), f'{assert_sms} method in ("3sigma", "Tukey")!'

        outliers = DataFrame()
        for col in self.select_dtypes(include='number').columns:
            lower_bound, upper_bound = -np.inf, np.inf
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

    def outliers(self, nu: float = 0.1):  # TODO: доделать
        models = [OneClassSVM(nu=nu),  # nu - % выбросов
                  IsolationForest(),
                  EllipticEnvelope(contamination=0.2),
                  LocalOutlierFactor(novelty=True)]
        for i, model in enumerate(models):
            model.fit(self)

    def find_corr_features(self, threshold: float = 0.85) -> list[str]:
        com_m = self.corr().abs()

        # верхний треугольник матрицы корреляции без диагонали
        uptriangle = com_m.where(np.triu(np.ones(com_m.shape), k=1).astype(bool))

        # индексы столбцов с корреляцией выше порога
        to_drop = [column for column in uptriangle.columns[1:]
                   if any(threshold <= uptriangle[column]) or all(uptriangle[column].isna())]

        return to_drop

    def l1_models(self, y, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False) -> list:
        """Линейные модели с разной L1-регуляризацией"""
        assert_sms = 'Incorrect assert:'
        assert type(y) is str, f'{assert_sms} type(y) is str'
        assert isiter(l1), f'{assert_sms} isiter(l1)'

        x = StandardScaler().fit_transform(self.drop([y], axis=1)) if scale else self.drop([y], axis=1)
        result = list()
        for alpha in tqdm(l1, desc='Fitting L1-models'):
            model = Lasso(alpha=alpha).fit(x, self[y])
            result.append(model)
            if early_stopping and all(map(lambda c: c == 0, model.coef_)): break
        return result

    def l1_importance(self, y, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False):
        # TODO: threshold l1
        """Коэффициенты признаков линейной моедли с L1-регуляризацией"""
        l1_models = self.l1_models(y, l1=l1, scale=scale, early_stopping=early_stopping)
        df = DataFrame([l1_model.coef_ for l1_model in l1_models], columns=self.drop([y], axis=1).columns)
        return DataFrame(pd.concat([pd.DataFrame({'L1': l1}), df], axis=1))  # .fillna(0.0)

    def l1_importance_plot(self, y, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False,
                           **kwargs):
        """Построение коэффициентов признаков линейных моделей с L1-регуляризацией"""
        df = self.l1_importance(y, l1=l1, scale=scale, early_stopping=early_stopping)
        x = df.pop('L1')

        plt.figure(figsize=kwargs.get('figsize', (12, 9)))
        plt.grid(kwargs.get('grid', True))
        for column in df.columns:
            plt.plot(x, df[column])
        plt.legend(df.columns, fontsize=12)
        plt.xlabel('L1', fontsize=14)
        plt.ylabel('coef', fontsize=14)
        plt.xlim([0, l1[-1]])
        plt.show()

    @ignore_warnings
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
            model.fit(self.drop([target], axis=1), self[target])
            result = permutation_importance(model, self.drop([target], axis=1), self[target])
            return pd.Series(result['importances_mean'], index=self.columns[:-1]).sort_values(ascending=False)
        except Exception as e:
            print(e)

    def permutation_importance_plot(self, target: str, **kwargs):
        """Перемешивающий подход на столбчатой диаграмме"""
        try:
            s = self.permutation_importance(target).sort_values(ascending=True)
        except:
            return
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.barh(s.index, s)
        plt.show()

    def feature_importances(self, target: str):
        """Важные признаки для классификации"""
        model = RandomForestClassifier()
        try:
            model.fit(self.drop([target], axis=1), self[target])
            return pd.Series(model.feature_importances_, index=self.columns[:-1]).sort_values(ascending=False)
        except Exception as e:
            print(e)

    def feature_importances_plot(self, target: str, **kwargs):
        """Важные признаки для классификации на столбчатой диаграмме"""
        try:
            s = self.feature_importances(target).sort_values(ascending=True)
        except:
            return
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.barh(s.index, s)
        plt.show()

    def balance(self, column_name):
        """Сбалансированность класса"""
        return self.groupby(column_name).count()  # TODO: подумать

    def corrplot(self, title='Correlation', fmt=3, **kwargs):
        """Тепловая карта матрицы корреляции"""
        plt.figure(figsize=kwargs.get('figsize', (12, 12)))
        plt.title(title, fontsize=16, fontweight='bold')
        sns.heatmap(self.corr(), annot=True, fmt=f'.{fmt}f')
        if kwargs.get('savefig', False): export2(plt, file_name=title, file_extension='png')

    def pairplot(self, **kwargs):
        sns.set(style='whitegrid')
        g = sns.PairGrid(self, diag_sharey=False, height=4)
        g.fig.set_size_inches(kwargs.get('figsize', (12, 12)))
        g.map_diag(sns.kdeplot, lw=2)
        g.map_lower(sns.scatterplot, s=25, edgecolor="k", linewidth=0.5, alpha=0.4)
        g.map_lower(sns.kdeplot, cmap='plasma', n_levels=6, alpha=0.5)
        plt.tight_layout()
        if kwargs.get('savefig', False): export2(plt, file_name='pair_plot', file_extension='png')

    def histplot(self, bins=40, **kwargs):
        self.hist(figsize=kwargs.get('figsize', (12, 12)), bins=bins)
        if kwargs.get('savefig', False): export2(plt, file_name='histplot', file_extension='png')

    def boxplot(self, title='boxplot', scale=False, fill=True, grid=True, **kwargs):
        plt.figure(figsize=kwargs.get('figsize', (12, 9)))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(grid)
        if not scale:
            sns.boxplot(self, fill=fill)
        else:
            sns.boxplot(pd.DataFrame(StandardScaler().fit_transform(self), columns=self.columns), fill=fill)
        if kwargs.get('savefig', False): export2(plt, file_name='boxplot', file_extension='png')

    def train_test_split(self, test_size, shuffle=True, random_state=0):
        """Разделение DataFrame на тренировочный и тестовый"""
        # stratify не нужен в виду разбиение одного датафрейма self
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

    def precision_recall_curve(self, y_true, y_predicted, title='precision recall curve', **kwargs):
        """График PR"""
        precision, recall, threshold = precision_recall_curve(y_true, y_predicted)
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.plot(precision, recall, color='blue', label='PR')
        plt.grid(True)
        plt.xlabel('precision', fontsize=14)
        plt.ylabel('recall', fontsize=14)
        plt.show()

    def roc_curve(self, y_true, y_predicted, title='roc curve', **kwargs):
        """График ROC"""
        fpr, tpr, threshold = roc_curve(y_true, y_predicted)
        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.title(title, fontsize=16, fontweight='bold')
        plt.plot(fpr, tpr, color='blue', label='ROC')
        plt.plot([0, 1], [0, 1], color='red')
        plt.grid(True)
        plt.xlabel('fpr', fontsize=14)
        plt.plot('tpr', fontsize=14)
        plt.show()

    def save(self, path: str) -> None:
        pickle.dump(self.__model, open(path, 'wb'))

    def load(self, path: str):
        self.__model = pickle.load(open(path, 'rb'))
        return self


def classifier_or_regressor(model) -> str:
    if 'cla' in (type(model).__name__.lower(), model.__name__.lower()): return 'cla'
    if 'reg' in (type(model).__name__.lower(), model.__name__.lower()): return 'reg'
    return ''


class Bagging:

    # TODO: доделать
    def __init__(self, model, n_estimators, max_samples, max_features, random_state=42):
        bagging_type = classifier_or_regressor(model)
        if bagging_type == 'cla':
            self.bagging = BaggingClassifier()
        elif bagging_type == 'reg':
            self.bagging = BaggingRegressor()
        else:
            raise '!!!'

        self.model = model
        self.n_estimators = n_estimators,
        self.max_samples = max_samples,
        self.max_features = max_features,
        self.random_state = random_state

    def fit(self, x, y):
        self.bagging.fit(x, y)


class Stacking:

    # TODO: доделать
    def __init__(self, model):
        self.stacking = StackingClassifier() if 'cla' in type(model).__name__.lower() else StackingRegressor()


class Boosting:

    # TODO: доделать
    def __init__(self):
        pass


if __name__ == '__main__':
    df = DataFrame(pd.read_csv('airfoil_self_noise.dat', sep="\t", header=None))
    df.columns = ["Frequency [Hz]", "Attack angle [deg]", "Chord length [m]", "Free-stream velocity [m/s]",
                  "Thickness [m]", "Pressure level [db]"]
    print(df)
    print(df.detect_outliers())
    print(df.find_corr_features())
    print(df.encode_one_hot(["Frequency [Hz]"]))

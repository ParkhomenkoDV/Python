from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy

# библиотеки визуализации
import matplotlib.pyplot as plt
import seaborn as sns

# библиотеки ML
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import (f_classif as f_classification, mutual_info_classif as mutual_info_classification,
                                       chi2)
from sklearn.feature_selection import (f_regression, mutual_info_regression)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFE as RecursiveFeatureElimination
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.preprocessing import (Normalizer,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler,
                                   RobustScaler, QuantileTransformer, PowerTransformer)

from sklearn.inspection import permutation_importance

# понижение размерности
from sklearn.decomposition import PCA as PrincipalComponentAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.manifold import TSNE as TDistributedStochasticNeighborEmbedding

# разделение данных
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit

# модели ML
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import mutual_info_score

from imblearn import under_sampling, over_sampling

'''import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

import pymorphy2 as pymorphy  # model = pymorphy.MorphAnalyzer'''

# частные библиотеки
import decorators
from tools import isiter, export2


class DataFrame(pd.DataFrame):
    """Расширенный класс pandas.DataFrame"""

    def __init__(self, *args, **kwargs):
        super(DataFrame, self).__init__(*args, **kwargs)
        self.__target = ''  # имя целевой колонки

    @property
    def target(self) -> str:
        return self.__target

    @target.setter
    def target(self, target: str):
        if target in self.columns:
            self.__target = target
        else:
            raise Exception(f'target "{self.__target}" not in {self.columns.to_list()}')

    @target.deleter
    def target(self):
        self.__target = ''

    def __get_target(self, **kwargs) -> str:
        """Получение target из словаря или приватного атрибута"""
        target = kwargs.get('target', self.__target)
        assert type(target) is str, f'{self.assert_sms} type(target) is str'
        assert target in self.columns, f'target "{self.__target}" not in {self.columns.to_list()}'
        return target

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

    def polynomial_features(self, columns: list, degree: int, include_bias=False):
        """Полиномирование признаков"""

        assert type(columns) is list, 'type(columns) is list'
        assert len(columns) > 0, 'len(columns) > 0'
        assert all(map(lambda col: col in self.columns, columns)), 'all(map(lambda col: col in self.columns, columns))'
        assert type(degree) is int, 'type(degree) is int'
        assert degree > 1, 'degree > 1'
        assert type(include_bias) is bool, 'type(include_bias) is bool'

        pf = PolynomialFeatures(degree=degree, include_bias=include_bias)
        df = DataFrame(pf.fit_transform(self[columns]), columns=pf.get_feature_names_out())
        return df

    def __assert_vectorize(self, **kwargs):
        """Проверка параметров векторизации"""

        # кодировка
        encoding = kwargs.get('encoding', 'utf-8')
        assert type(encoding) is str, 'type(encoding) is str'

        # перевод токенов в нижний регистр
        lowercase = kwargs.get('lowercase', True)
        assert type(lowercase) is bool, 'type(lowercase) is bool'

        # учет стоп-слов
        stop_words = kwargs.get('stop_words', None)
        assert stop_words is None or type(stop_words) is list, 'stop_words is None or type(stop_words) is list'
        if type(stop_words) is list:
            assert all(map(lambda w: type(w) is str, stop_words)), 'all(map(lambda w: type(w) is str, stop_words))'

        # пределы слов в одном токене
        ngram_range = kwargs.get('ngram_range', (1, 1))
        assert type(ngram_range) in (tuple, list), 'type(ngram_range) in (tuple, list)'
        assert len(ngram_range) == 2, 'len(ngram_range) == 2'
        assert all(map(lambda x: type(x) is int, ngram_range)), 'all(map(lambda x: type(x) is int, ngram_range))'
        assert ngram_range[0] <= ngram_range[1], 'ngram_range[0] <= ngram_range[1]'

        # анализатор разбиения
        analyzer = kwargs.get('analyzer', 'word')
        assert type(analyzer) is str, 'type(analyzer) is str'
        analyzer = analyzer.strip().lower()
        assert analyzer in ("word", "char", "char_wb"), 'analyzer in ("word", "char", "char_wb")'

    def vectorize_count(self, columns: list[str], drop=False, inplace=False, **kwargs):
        """Количественная векторизация токенов"""

        self.__assert_vectorize(**kwargs)

        corpus = self[columns].to_numpy().flatten()
        vectorizer = CountVectorizer(**kwargs)
        df = DataFrame(vectorizer.fit_transform(corpus).toarray(), columns=vectorizer.get_feature_names_out())

        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def vectorize_tf_idf(self, columns: list[str], drop=False, inplace=False, **kwargs):
        """tf-idf векторизация токенов"""

        self.__assert_vectorize(**kwargs)

        corpus = self[columns].to_numpy().flatten()
        vectorizer = TfidfVectorizer(**kwargs)
        df = DataFrame(vectorizer.fit_transform(corpus).toarray(), columns=vectorizer.get_feature_names_out())

        if drop: self.__init__(self.drop(columns, axis=1))
        if inplace:
            self.__init__(pd.concat([self, df], axis=1))
        else:
            return df

    def fill_na(self, value=None, method='', inplace=False):
        """Заполнение nan, null, na значений в DataFrame согласно значению value или методу method"""

        assert value is None or type(value) in (int, float, str)
        assert type(method) is str
        method = method.strip().lower()

        if value is not None:
            df = self.fillna(value)
        else:
            match method:
                case 'mean':
                    df = self.fillna(self.mean())
                case 'median':
                    df = self.fillna(self.median())
                case 'mode':
                    df = self.fillna(self.mode())
                case 'hmean':
                    df = self.fillna(pd.Series([scipy.stats.hmean(col[~np.isnan(col)]) for col in self.values.T],
                                               index=self.columns))
                case 'indicator':
                    temple_df = pd.DataFrame(self.isna().astype(int).to_numpy(),
                                             columns=self.isna().columns + '_na')
                    df = pd.concat((self, temple_df), axis=1)
                    df = df.loc[:, (df != 0).any(axis=0)]
                case 'interpolation':
                    df = self.interpolate(method='linear', limit_direction='forward')
                    df = self.fillna(df)
                case 'prev':
                    df = self.fillna(self.ffill()).fillna(self.bfill())
                case 'next':  # TODO
                    pass
                case 'model':  # TODO
                    pass
                case _:
                    valid_methods = ('mean', 'median', 'mode', 'hmean', 'indicator', 'prev_num', 'interpolation')
                    raise ValueError(f"Incorrect method. This method doesn't support. Valid methods: {valid_methods}")

        if inplace:
            self.__init__(df)
            return
        else:
            return df

    def distribution(self, columns: list[str] | tuple[str]) -> dict[str:dict[str:float]]:
        """Определение распределения атрибутов"""
        assert type(columns) in (list, tuple)
        assert all(map(lambda column: column in self.columns, columns))

        result = dict()
        for column in columns:
            skew, kurtosis = self[column].skew(), self[column].kurtosis()
            result[column] = {'skew': skew, 'kurtosis': kurtosis, 'normal': abs(skew) <= 2 and abs(kurtosis) <= 7}
        return result

    def variation_coefficient(self, columns: list[str] | tuple[str]) -> dict[str:float]:
        """Коэффициент вариации = мера разброса данных"""
        assert type(columns) in (list, tuple)
        assert all(map(lambda column: column in self.columns, columns))
        return {column: self[column].std() / self[column].mean() for column in columns}

    def detect_outliers(self, method: str = 'sigma', **kwargs):
        """Обнаружение выбросов статистическим методом"""
        assert type(method) is str, f'type(method) is str'
        method = method.strip().lower()

        outliers = DataFrame()
        for col in self.select_dtypes(include='number').columns:
            lower_bound, upper_bound = -np.inf, np.inf
            if method == 'sigma':  # если данные распределены нормально!
                mean = self[col].mean()
                std = self[col].std()
                lower_bound, upper_bound = mean - kwargs.get('k', 3) * std, mean + kwargs.get('k', 3) * std
            elif method == 'tukey':
                q1, q3 = self[col].quantile(0.25), self[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - kwargs.get('k', 1.5) * iqr, q3 + kwargs.get('k', 1.5) * iqr
            elif method == 'shovene':
                mean = self[col].mean()
                std = self[col].std()
                d = self[scipy.special.erfc(abs(self[col] - mean) / std) < 1 / (2 * len(self[col]))]
                outliers = pd.concat([outliers, d]).drop_duplicates().sort_index()
            else:
                raise Exception('method in ("Sigma", "Tukey", "Shovene")')

            col_outliers = self[(self[col] < lower_bound) | (self[col] > upper_bound)]
            outliers = pd.concat([outliers, col_outliers])
        return DataFrame(outliers).drop_duplicates().sort_index()

    def detect_model_outliers(self, fraction: float, threshold=0.5, **kwargs):
        """Обнаружение выбросов модельным методом"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(fraction) is float, 'type(fraction) is float'
        assert 0 < fraction <= 0.5, '0 < fraction <= 0.5'  # 'contamination' param EllipticEnvelope in (0.0, 0.5]
        assert type(threshold) is float, 'type(threshold) is float'

        models = [OneClassSVM(nu=fraction),  # fraction - доля выбросов
                  IsolationForest(),
                  EllipticEnvelope(contamination=fraction),  # fraction - доля выбросов
                  LocalOutlierFactor(novelty=True)]  # для новых данных

        outliers = DataFrame()
        for model in models:  # для каждой модели
            model.fit(x.values, y)  # обучаем (.values необходим для анонимности данных)
            pred = model.predict(x.values)  # предсказываем (.values необходим для анонимности данных)
            pred[pred == -1] = False  # выбросы (=-1) переименуем в False (=0)
            pred = DataFrame(pred, columns=[model.__class__.__name__])  # создаем DataFrame
            outliers = pd.concat([outliers, pred], axis=1)  # конкатезируем выбросы по данной модели

        # вероятность НЕ выброса (адекватных данных) определяется как среднее арифметическое предсказаний всех моделей
        outliers['probability'] = outliers.apply(lambda row: row.mean(), axis=1)
        # выброс считается, когда вероятность адекватных данных < 1 - порог вероятности выброса
        outliers['outlier'] = outliers['probability'] < (1 - threshold)
        return self[outliers['outlier']].sort_index()

    def corr_features(self, method='pearson', threshold: float = 0.85) -> dict[tuple[str]:float]:
        """Линейно-независимые признаки"""
        assert type(method) is str
        method = method.strip().lower()
        assert method in ('pearson', 'kendall', 'spearman')

        assert type(threshold) is float
        assert -1 <= threshold <= 1

        corr_matrix = self.corr(method=method).abs()  # матрица корреляции

        # верхний треугольник матрицы корреляции без диагонали
        uptriangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # группировка столбцов по столбцам и сортировка по убыванию значений линейной зависимости
        result = uptriangle.unstack(-1).sort_values(ascending=False)

        return result[result >= threshold].to_dict()

    def l1_models(self, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False, **kwargs) -> list:
        """Линейные модели с разной L1-регуляризацией"""
        target = self.__get_target(**kwargs)
        assert isiter(l1), f'{self.assert_sms} isiter(l1)'
        assert all(isinstance(el, (float, int)) for el in l1), \
            f'{self.assert_sms} all(isinstance(el, (float, int)) for el in l1)'

        x, y = self.feature_target_split(target=target)
        x = StandardScaler().fit_transform(x) if scale else x

        result = list()
        for alpha in tqdm(l1, desc='Fitting L1-models'):
            model = Lasso(alpha=alpha).fit(x, y)
            result.append(model)
            if early_stopping and all(map(lambda c: c == 0, model.coef_)): break
        return result

    def l1_importance(self, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False, **kwargs):
        """Коэффициенты признаков линейной моедли с L1-регуляризацией"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        l1_models = self.l1_models(l1=l1, scale=scale, early_stopping=early_stopping, target=target)

        df = DataFrame([l1_model.coef_ for l1_model in l1_models], columns=x.columns)
        return DataFrame(pd.concat([pd.DataFrame({'L1': l1}), df], axis=1))

    def l1_importance_plot(self, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False, **kwargs):
        """Построение коэффициентов признаков линейных моделей с L1-регуляризацией"""
        target = self.__get_target(**kwargs)

        df = self.l1_importance(l1=l1, scale=scale, early_stopping=early_stopping, target=target)
        df.dropna(axis=0, inplace=True)
        x = df.pop('L1')

        plt.figure(figsize=kwargs.get('figsize', (12, 9)))
        plt.grid(kwargs.get('grid', True))
        for column in df.columns:
            plt.plot(x, df[column])
        plt.legend(df.columns, fontsize=12)
        plt.xlabel('L1', fontsize=14)
        plt.ylabel('coef', fontsize=14)
        plt.xlim([0, l1[x.shape[0]]])
        plt.show()

    def select_l1_features(self, n_features: int, **kwargs) -> list[str]:
        """Выбор n_features штук features с весомыми коэффициентами L1-регуляризации"""
        assert type(n_features) is int, f'{self.assert_sms} type(n_features) is int'
        assert 1 <= n_features < len(self.columns), f'{self.assert_sms} 1 <= n_features < len(self.columns)'

        l1_importance = self.l1_importance(**kwargs).drop(['L1'], axis=1).dropna(axis=0)

        l1_features = list()
        for row in (l1_importance != 0)[::-1].to_numpy():
            if row.sum() >= n_features:
                l1_features = l1_importance.columns[row].to_list()
                break
        else:
            return l1_importance.columns[l1_importance.iloc[-1] != 0].to_list()

        return l1_features

    @decorators.warns('ignore')
    def mutual_info_score(self, **kwargs):
        """Взаимная информация корреляции"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        result = dict()
        for column in x:
            result[column] = mutual_info_score(x[column], y)
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return pd.Series(dict(result))

    def mutual_info_score_plot(self, **kwargs):
        """График взаимной информации корреляции"""
        mutual_info_score = self.mutual_info_score(**kwargs).sort_values(ascending=True)

        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.xlabel('mutual info score')
        plt.ylabel('features')
        plt.barh(mutual_info_score.index, mutual_info_score)
        plt.show()

    def select_mutual_info_score_features(self, threshold: int | float, **kwargs) -> list[str]:
        """Выбор признаков по взаимной информации корреляции"""
        mutual_info_score_features = self.mutual_info_score(**kwargs)
        if type(threshold) is int:  # количество выбираемых признаков
            assert 1 <= threshold <= len(mutual_info_score_features), \
                f'{self.assert_sms} 1 <= threshold <= {len(mutual_info_score_features)}'
            return mutual_info_score_features[:threshold].index.to_list()
        elif type(threshold) is float:  # порог значения признаков
            assert 0 < threshold, f'{self.assert_sms} 0 < threshold'
            return mutual_info_score_features[mutual_info_score_features > threshold].index.to_list()
        else:
            raise Exception(f'{self.assert_sms} type(threshold) in (int, float)')

    def permutation_importance(self, **kwargs):
        """Перемешивающий метод"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        for model in (RandomForestClassifier(), RandomForestRegressor()):
            try:
                model.fit(x, y)
            except Exception as exception:
                continue
            result = permutation_importance(model, x, y)
            return pd.Series(result['importances_mean'], index=x.columns).sort_values(ascending=False)

    def permutation_importance_plot(self, **kwargs):
        """Перемешивающий метод на столбчатой диаграмме"""
        permutation_importance = self.permutation_importance(**kwargs).sort_values(ascending=True)

        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.barh(permutation_importance.index, permutation_importance)
        plt.show()

    def select_permutation_importance_features(self, threshold: int | float, **kwargs) -> list[str]:
        """Выбор признаков перемешивающим методом"""
        permutation_importance_features = self.permutation_importance(**kwargs)
        if type(threshold) is int:  # количество выбираемых признаков
            assert 1 <= threshold <= len(permutation_importance_features), \
                f'{self.assert_sms} 1 <= threshold <= {len(permutation_importance_features)}'
            return permutation_importance_features[:threshold].index.to_list()
        elif type(threshold) is float:  # порог значения признаков
            assert 0 < threshold, f'{self.assert_sms} 0 < threshold'
            return permutation_importance_features[permutation_importance_features > threshold].index.to_list()
        else:
            raise Exception(f'{self.assert_sms} type(threshold) in (int, float)')

    def random_forest_importance_features(self, **kwargs):
        """Важные признаки случайного леса"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)
        for model in (RandomForestClassifier(), RandomForestRegressor()):
            try:
                model.fit(x, y)
            except Exception as exception:
                continue
            return pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)

    def random_forest_importance_features_plot(self, **kwargs):
        """Важные признаки для классификации на столбчатой диаграмме"""
        target = self.__get_target(**kwargs)
        importance_features = self.random_forest_importance_features(target=target).sort_values(ascending=True)

        plt.figure(figsize=kwargs.get('figsize', (9, 9)))
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.barh(importance_features.index, importance_features)
        plt.show()

    def select_random_forest_importance_features(self, threshold: int | float, **kwargs) -> list[str]:
        """Выбор важных признаков для классификации"""
        importance_features = self.random_forest_importance_features(**kwargs)
        if type(threshold) is int:  # количество выбираемых признаков
            assert 1 <= threshold <= len(importance_features), \
                f'{self.assert_sms} 1 <= threshold <= {len(importance_features)}'
            return importance_features[:threshold].index.to_list()
        elif type(threshold) is float:  # порог значения признаков
            assert 0 < threshold, f'{self.assert_sms} 0 < threshold'
            return importance_features[importance_features > threshold].index.to_list()
        else:
            raise Exception(f'{self.assert_sms} type(threshold) in (int, float)')

    def __select_metric(self, metric: str):
        """Вспомогательная функция к выбору метрики"""
        METRICS = {'classification': ('f_classification', 'mutual_info_classification', 'chi2'),
                   'regression': ('f_regression', 'mutual_info_regression')}

        assert type(metric) is str, f'{self.assert_sms} type(metrics) is str'
        metric = metric.strip().lower()
        assert metric in [v for value in METRICS.values() for v in value], f'{self.assert_sms} metrics in {METRICS}'

        if metric == 'f_classification': return f_classification
        if metric == 'mutual_info_classification': return mutual_info_classification
        if metric == 'chi2': return chi2
        if metric == 'f_regression': return f_regression
        if metric == 'mutual_info_regression': return mutual_info_regression

    def select_k_best_features(self, metric: str, k: int, inplace=False, **kwargs):
        """Выбор k лучших признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(k) is int, f'{self.assert_sms} type(k) is int'
        assert 1 <= k <= len(x.columns), f'{self.assert_sms} 1 <= k <= {len(x.columns)}'

        skb = SelectKBest(self.__select_metric(metric), k=k)
        x_reduced = DataFrame(skb.fit_transform(x, y), columns=x.columns[skb.get_support()])

        x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y, # ломает регрессию
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=True, random_state=0)
        if kwargs.get('test_size', None):
            for model in (KNeighborsClassifier(), KNeighborsRegressor()):
                try:
                    model.fit(x_train, y_train)
                except Exception as exception:
                    continue
                score = model.score(x_test, y_test)
                print(f'score: {score}')

        if inplace:
            self.__init__(x_reduced)
        else:
            return x.columns[skb.get_support()].to_list()

    def select_percentile_features(self, metric: str, percentile: int | float, inplace=False, **kwargs):
        """Выбор указанного процента признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(percentile) in (int, float), f'{self.assert_sms} type(percentile) in (int, float)'
        assert 0 < percentile < 100, f'{self.assert_sms} 0 < percentile < 100'

        sp = SelectPercentile(self.__select_metric(metric), percentile=percentile)
        x_reduced = DataFrame(sp.fit_transform(x, y), columns=x.columns[sp.get_support()])

        x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y, # ломает регрессию
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=True, random_state=0)
        if kwargs.get('test_size', None):
            for model in (KNeighborsClassifier(), KNeighborsRegressor()):
                try:
                    model.fit(x_train, y_train)
                except Exception as exception:
                    continue
                score = model.score(x_test, y_test)
                print(f'score: {score}')

        if inplace:
            self.__init__(x_reduced)
        else:
            return x_reduced

    def select_elimination_features(self, n_features_to_select: int, step: int, inplace=False, **kwargs):
        """Выбор n_features_to_select лучших признаков, путем рекурсивного удаления худших по step штук за итерацию"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(n_features_to_select) is int, f'{self.assert_sms} type(n_features_to_select) is int'
        assert 1 <= n_features_to_select, f'{self.assert_sms} 1 <= n_features_to_select'
        assert type(step) is int, f'{self.assert_sms} type(step) is int'
        assert 1 <= step, f'{self.assert_sms} 1 <= step'

        for model in (RandomForestClassifier(), RandomForestRegressor()):
            rfe = RecursiveFeatureElimination(model, n_features_to_select=n_features_to_select, step=step)
            try:
                x_reduced = DataFrame(rfe.fit_transform(x, y), columns=x.columns[rfe.get_support()])
            except Exception as exception:
                continue

            x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                                test_size=kwargs.get('test_size', 0.25),
                                                                shuffle=True, random_state=0)
            if kwargs.get('test_size', None):
                model.fit(x_train, y_train)
                score = model.score(x_test, y_test)
                print(f'score: {score}')
                # print(rfe.support_)  # кто удалился
                # print(rfe.ranking_)  # порядок удаления features (кто больше, тот раньше ушел)

            if inplace:
                self.__init__(x_reduced)
                return
            else:
                return x_reduced

    def select_from_model_features(self, max_features: int, threshold=-np.inf, inplace=False, **kwargs):
        """Выбор важных для классификации признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(max_features) is int, f'{self.assert_sms} type(max_features) is int'

        for model in (RandomForestClassifier(), RandomForestRegressor()):
            sfm = SelectFromModel(model, prefit=False, max_features=max_features, threshold=threshold)
            try:
                x_reduced = DataFrame(sfm.fit_transform(x, y), columns=x.columns[sfm.get_support()])
            except Exception as exception:
                continue

            x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                                test_size=kwargs.get('test_size', 0.25),
                                                                shuffle=True, random_state=0)
            if kwargs.get('test_size', None):
                model.fit(x_train, y_train)
                score = model.score(x_test, y_test)
                print(f'score: {score}')

            if inplace:
                self.__init__(x_reduced)
            else:
                return x_reduced

    def select_sequential_features(self, n_features_to_select: int, direction: str, inplace=False, **kwargs):
        """Последовательный выбор признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(n_features_to_select) is int, f'{self.assert_sms} type(n_features_to_select) is int'
        assert 1 < n_features_to_select, f'{self.assert_sms} 1 < n_features_to_select'
        assert direction in ("forward", "backward"), f'{self.assert_sms} direction in ("forward", "backward")'

        for model in (RandomForestClassifier(), RandomForestRegressor()):
            sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction)
            try:
                x_reduced = DataFrame(sfs.fit_transform(x, y), columns=x.columns[sfs.get_support()])
            except Exception as exception:
                continue
            x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                                test_size=kwargs.get('test_size', 0.25),
                                                                shuffle=True, random_state=0)
            if kwargs.get('test_size', None):
                model.fit(x_train, y_train)
                score = model.score(x_test, y_test)
                print(f'score: {score}')

            if inplace:
                self.__init__(x_reduced)
            else:
                return x_reduced

    def balance(self, column_name: str, threshold: int | float):
        """Сбалансированность класса"""
        assert column_name in self.columns, f'{self.assert_sms} column_name in {self.columns}'
        assert type(threshold) in (int, float), f'{self.assert_sms} type(threshold) in (int, float)'
        assert 1 < threshold, f'{self.assert_sms} 1 < threshold'

        df = self.value_counts(column_name).to_frame()
        df['fraction'] = df['count'] / len(self)
        df['balance'] = df['count'].max() / df['count'].min() <= threshold
        '''
        если отношение количества значений самого многочисленного класса 
        к количеству значений самого малочисленного класса 
        меньше или равно threshold раз, то баланс есть
        '''
        return DataFrame(df)

    def undersampling(self, target: str | int | float, method='RandomUnderSampler', **kwargs):
        """Under sampling dataset according to method

        Supported methods:
        'RandomUnderSampler',
        'EEditedNearestNeighbours',
        'RepeatedEditedNearestNeighbours',
        'AllKNN'
        'CondensedNearestNeighbour',
        'OneSidedSelection',
        'NeighbourhoodCleaningRule',
        'ClusterCentroids',
        'TomekLinks',
        'NearMiss',
        'InstanceHardnessThreshold'

        Parameters:
        df (pd.DataFrame): Input dataset
        target (str | int | float): Name of the target column
        method (str): Under-sampling method (default='random')
        **kwargs: Additional parameters for the under-sampling method

        Returns:
        pd.DataFrame: Under-sampled dataset
        """
        # checking common input parameters
        valid_methods = ('RandomUnderSampler', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN',
                         'CondensedNearestNeighbour', 'OneSidedSelection', 'NeighbourhoodCleaningRule',
                         'ClusterCentroids',
                         'TomekLinks', 'NearMiss', 'InstanceHardnessThreshold')

        assert method in valid_methods, f"This method doesn't support. Valid methods: {valid_methods}"
        assert target in df.columns, f'target must be in columns of df: {list(df.columns)}'

        # listing all possible parameters for sample processing
        random_state = kwargs.pop('random_state', None)
        sampling_strategy = kwargs.pop('sampling_strategy', 'auto')
        n_neighbors = kwargs.pop('n_neighbors', 3)

        kind_sel = kwargs['kind_sel'] if 'kind_sel' in kwargs.keys() else 'all'
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs.keys() else None
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs.keys() else 100
        allow_minority = kwargs['allow_minority'] if 'allow_minority' in kwargs.keys() else False
        n_seeds_S = kwargs['n_seeds_S'] if 'n_seeds_S' in kwargs.keys() else 1
        edited_nearest_neighbours = kwargs[
            'edited_nearest_neighbours'] if 'edited_nearest_neighbours' in kwargs.keys() else None
        threshold_cleaning = kwargs['threshold_cleaning'] if 'threshold_cleaning' in kwargs.keys() else 0.5
        estimator = kwargs['estimator'] if 'estimator' in kwargs.keys() else None
        voting = kwargs['voting'] if 'voting' in kwargs.keys() else 'auto'
        version = kwargs['version'] if 'version' in kwargs.keys() else 1
        n_neighbors_ver3 = kwargs['n_neighbors_ver3'] if 'n_neighbors_ver3' in kwargs.keys() else 3
        cv = kwargs['cv'] if 'cv' in kwargs.keys() else 5

        # main program
        match method:
            case 'RandomUnderSampler':
                sampler = under_sampling.RandomUnderSampler(random_state=random_state,
                                                            sampling_strategy=sampling_strategy)
            case 'EditedNearestNeighbours':
                sampler = under_sampling.EditedNearestNeighbours(sampling_strategy=sampling_strategy,
                                                                 n_neighbors=n_neighbors,
                                                                 kind_sel=kind_sel,
                                                                 n_jobs=n_jobs)
            case 'RepeatedEditedNearestNeighbours':
                sampler = under_sampling.RepeatedEditedNearestNeighbours(sampling_strategy=sampling_strategy,
                                                                         n_neighbors=n_neighbors,
                                                                         max_iter=max_iter,
                                                                         kind_sel=kind_sel,
                                                                         n_jobs=n_jobs)
            case 'AllKNN':
                sampler = under_sampling.AllKNN(sampling_strategy=sampling_strategy,
                                                n_neighbors=n_neighbors,
                                                allow_minority=allow_minority,
                                                kind_sel=kind_sel,
                                                n_jobs=n_jobs)
            case 'CondensedNearestNeighbour':
                sampler = under_sampling.CondensedNearestNeighbour(sampling_strategy=sampling_strategy,
                                                                   random_state=random_state,
                                                                   n_neighbors=n_neighbors,
                                                                   n_jobs=n_jobs,
                                                                   n_seeds_S=n_seeds_S)
            case 'OneSidedSelection':
                sampler = under_sampling.OneSidedSelection(sampling_strategy=sampling_strategy,
                                                           random_state=random_state,
                                                           n_neighbors=n_neighbors,
                                                           n_jobs=n_jobs,
                                                           n_seeds_S=n_seeds_S)
            case 'NeighbourhoodCleaningRule':
                sampler = under_sampling.NeighbourhoodCleaningRule(sampling_strategy=sampling_strategy,
                                                                   edited_nearest_neighbours=edited_nearest_neighbours,
                                                                   n_neighbors=n_neighbors,
                                                                   n_jobs=n_jobs,
                                                                   kind_sel=kind_sel,
                                                                   threshold_cleaning=threshold_cleaning)
            case 'ClusterCentroids':
                sampler = under_sampling.ClusterCentroids(sampling_strategy=sampling_strategy,
                                                          random_state=random_state,
                                                          estimator=estimator,
                                                          voting=voting)
            case 'TomekLinks':
                sampler = under_sampling.TomekLinks(sampling_strategy=sampling_strategy,
                                                    n_jobs=n_jobs)
            case 'NearMiss':
                sampler = under_sampling.NearMiss(sampling_strategy=sampling_strategy,
                                                  version=version,
                                                  n_neighbors=n_neighbors,
                                                  n_neighbors_ver3=n_neighbors_ver3,
                                                  n_jobs=n_jobs)
            case 'InstanceHardnessThreshold':
                sampler = under_sampling.InstanceHardnessThreshold(estimator=estimator,
                                                                   sampling_strategy=sampling_strategy,
                                                                   random_state=random_state,
                                                                   cv=cv,
                                                                   n_jobs=n_jobs)

        changed_data, changed_labels = sampler.fit_resample(df.to_numpy(), df[target].to_numpy())
        return pd.DataFrame(changed_data, changed_labels, columns=df.columns)

    def oversampling(self, target: str | int | float, method='RandomOverSampler', **kwargs):
        """
        Over sampling dataset according to method

        Supported methods:
        'RandomOverSampler',
        'SMOTE',
        'ADASYN',
        'BorderlineSMOTE',
        'SVMSMOTE',
        'KMeansSMOTE',
        'SMOTENC',
        'SMOTEN'

        Parameters:
        df (pd.DataFrame): Input dataset
        target (str | int | float): Name of the target column
        method (str): Over-sampling method (default='RandomOverSampler')
        **kwargs: Additional parameters for the over-sampling method

        Returns:
        pd.DataFrame: Over-sampled dataset
        """
        # checking common input parameters
        valid_methods = (
            'RandomOverSampler', 'SMOTE', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSMOTE', 'SMOTENC', 'SMOTEN')
        assert isinstance(df, pd.DataFrame), f'Incorrect dtype. df: {type(df)} instead of {pd.DataFrame}'
        assert method in valid_methods, f"This method doesn't support. Valid methods: {valid_methods}"
        assert target in df.columns, f'target must be in columns of df: {list(df.columns)}'

        # listing all possible parameters for sample processing
        default_k_neighbors = 2 if method == 'KMeansSMOTE' else 5  # it's necessary because
        # there are several parameters with the same names and different default values (KMeansSMOTE=2, others=5)
        random_state = kwargs['random_state'] if 'random_state' in kwargs.keys() else None
        sampling_strategy = kwargs['sampling_strategy'] if 'sampling_strategy' in kwargs.keys() else 'auto'
        density_exponent = kwargs['density_exponent'] if 'density_exponent' in kwargs.keys() else 'auto'
        cluster_balance_threshold = kwargs[
            'cluster_balance_threshold'] if 'cluster_balance_threshold' in kwargs.keys() else 'auto'
        shrinkage = kwargs['shrinkage'] if 'shrinkage' in kwargs.keys() else None
        k_neighbors = kwargs['k_neighbors'] if 'k_neighbors' in kwargs.keys() else default_k_neighbors
        n_neighbors = kwargs['n_neighbors'] if 'n_neighbors' in kwargs.keys() else 5
        m_neighbors = kwargs['m_neighbors'] if 'm_neighbors' in kwargs.keys() else 10
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs.keys() else None
        kmeans_estimator = kwargs['kmeans_estimator'] if 'kmeans_estimator' in kwargs.keys() else None
        svm_estimator = kwargs['svm_estimator'] if 'svm_estimator' in kwargs.keys() else None  #
        out_step = kwargs['out_step'] if 'out_step' in kwargs.keys() else 0.5
        kind = kwargs['kind'] if 'kind' in kwargs.keys() else 'borderline-1'
        categorical_features = kwargs['categorical_features'] if 'categorical_features' in kwargs.keys() else None
        categorical_encoder = kwargs['categorical_encoder'] if 'categorical_encoder' in kwargs.keys() else None

        # main program
        match method:
            case 'RandomOverSampler':
                sampler = over_sampling.RandomOverSampler(random_state=random_state,
                                                          sampling_strategy=sampling_strategy,
                                                          shrinkage=shrinkage)
            case 'SMOTE':
                sampler = over_sampling.SMOTE(random_state=random_state,
                                              sampling_strategy=sampling_strategy,
                                              k_neighbors=k_neighbors,
                                              n_jobs=n_jobs)
            case 'ADASYN':
                sampler = over_sampling.ADASYN(random_state=random_state,
                                               sampling_strategy=sampling_strategy,
                                               n_neighbors=n_neighbors,
                                               n_jobs=n_jobs)
            case 'BorderlineSMOTE':
                sampler = over_sampling.BorderlineSMOTE(random_state=random_state,
                                                        sampling_strategy=sampling_strategy,
                                                        k_neighbors=k_neighbors,
                                                        m_neighbors=m_neighbors,
                                                        kind=kind,
                                                        n_jobs=n_jobs)
            case 'SVMSMOTE':
                sampler = over_sampling.SVMSMOTE(random_state=random_state,
                                                 sampling_strategy=sampling_strategy,
                                                 k_neighbors=k_neighbors,
                                                 m_neighbors=m_neighbors,
                                                 svm_estimator=svm_estimator,
                                                 out_step=out_step,
                                                 n_jobs=n_jobs)
            case 'KMeansSMOTE':
                sampler = over_sampling.KMeansSMOTE(random_state=random_state,
                                                    sampling_strategy=sampling_strategy,
                                                    k_neighbors=k_neighbors,
                                                    n_jobs=n_jobs,
                                                    kmeans_estimator=kmeans_estimator,
                                                    cluster_balance_threshold=cluster_balance_threshold,
                                                    density_exponent=density_exponent)
            case 'SMOTENC':
                sampler = over_sampling.SMOTENC(random_state=random_state,
                                                sampling_strategy=sampling_strategy,
                                                k_neighbors=k_neighbors,
                                                categorical_features=categorical_features,
                                                n_jobs=n_jobs,
                                                categorical_encoder=categorical_encoder)
                changed_data, changed_labels = sampler.fit_resample(df, df[target])
                return pd.DataFrame(changed_data, changed_labels, columns=df.columns)
            case 'SMOTEN':
                sampler = over_sampling.SMOTEN(random_state=random_state,
                                               sampling_strategy=sampling_strategy,
                                               k_neighbors=k_neighbors,
                                               n_jobs=n_jobs,
                                               categorical_encoder=categorical_encoder)

        changed_data, changed_labels = sampler.fit_resample(df.to_numpy(), df[target].to_numpy())
        return pd.DataFrame(changed_data, changed_labels, columns=df.columns)

    @decorators.try_except('pass')
    def pca(self, n_components: int, inplace=False, **kwargs):
        """Метод главный компонент для линейно-зависимых признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(n_components) is int, f'{self.assert_sms} type(n_components) is int'
        assert 1 <= n_components <= min(len(x.columns), len(y.unique())), \
            f'1 <= n_components <= {min(len(x.columns), len(y.unique()))}'

        pca = PrincipalComponentAnalysis(n_components=n_components)
        x_reduced = DataFrame(pca.fit_transform(x, y))
        print(f'Объем сохраненной и потерянной информации: {pca.explained_variance_ratio_}')  # потеря до 20% приемлема
        x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=True, random_state=0)
        if kwargs.get('test_size', None):
            model = KNeighborsClassifier().fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print(f'score: {score}')

        if inplace:
            self.__init__(x_reduced)
        else:
            return x_reduced

    @decorators.try_except('pass')
    def lda(self, n_components: int, inplace=False, **kwargs):
        """Линейно дискриминантный анализ для линейно-зависимых признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(n_components) is int, f'{self.assert_sms} type(n_components) is int'
        assert 1 <= n_components <= min(len(x.columns), len(y.unique()) - 1), \
            f'1 <= n_components <= {min(len(x.columns), len(y.unique()) - 1)}'

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        x_reduced = DataFrame(lda.fit_transform(x, y))
        x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=True, random_state=0)
        if kwargs.get('test_size', None):
            model = KNeighborsClassifier().fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print(f'score: {score}')

        if inplace:
            self.__init__(x_reduced)
        else:
            return x_reduced

    @decorators.try_except('pass')
    def nca(self, n_components: int, inplace=False, **kwargs):
        """Анализ компонентов соседств для нелинейных признаков"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(n_components) is int, f'{self.assert_sms} type(n_components) is int'
        assert 1 <= n_components <= min(len(x.columns), len(y.unique())), \
            f'1 <= n_components <= {min(len(x.columns), len(y.unique()))}'

        nca = NeighborhoodComponentsAnalysis(n_components=n_components)
        x_reduced = DataFrame(nca.fit_transform(x, y))
        x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=True, random_state=0)
        if kwargs.get('test_size', None):
            model = KNeighborsClassifier().fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print(f'score: {score}')

        if inplace:
            self.__init__(x_reduced)
        else:
            return x_reduced

    @decorators.try_except('pass')
    def tsne(self, n_components: int, inplace=False, **kwargs):
        """Стохастическое вложение соседей с t-распределением"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        assert type(n_components) is int, f'{self.assert_sms} type(n_components) is int'
        assert 1 <= n_components < min(len(x.columns), len(y.unique())), \
            f'1 <= n_components < {min(len(x.columns), len(y.unique()))}'

        perplexity = kwargs.get('perplexity', 30)  # сложность
        assert type(perplexity) in (int, float), f'type(perplexity) in (int, float)'
        assert 1 <= perplexity, '1 <= perplexity'

        tsne = TDistributedStochasticNeighborEmbedding(n_components=n_components, **kwargs)
        x_reduced = DataFrame(tsne.fit_transform(x, y))
        x_train, x_test, y_train, y_test = train_test_split(x_reduced, y,  # stratify=y,  # ломает регрессию
                                                            test_size=kwargs.get('test_size', 0.25),
                                                            shuffle=True, random_state=0)

        if kwargs.get('test_size', None):
            model = KNeighborsClassifier().fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print(f'score: {score}')

        if inplace:
            self.__init__(x_reduced)
        else:
            return x_reduced

    def corrplot(self, fmt=3, **kwargs):
        """Тепловая карта матрицы корреляции"""
        plt.figure(figsize=kwargs.get('figsize', (12, 12)))
        plt.title(kwargs.get('title', 'corrplot'), fontsize=16, fontweight='bold')
        sns.heatmap(self.corr(), annot=True, fmt=f'.{fmt}f')
        if kwargs.get('savefig', False): export2(plt, file_name=kwargs.get('title', 'corrplot'), file_extension='png')

    def pairplot(self, **kwargs):
        sns.set(style='whitegrid')
        g = sns.PairGrid(self, diag_sharey=False, height=4)
        g.fig.set_size_inches(kwargs.get('figsize', (12, 12)))
        g.map_diag(sns.kdeplot, lw=2)
        g.map_lower(sns.scatterplot, s=25, edgecolor="k", linewidth=0.5, alpha=0.4)
        g.map_lower(sns.kdeplot, cmap='plasma', n_levels=6, alpha=0.5)
        plt.tight_layout()
        if kwargs.get('savefig', False): export2(plt, file_name=kwargs.get('title', 'pairplot'), file_extension='png')

    def jointplot(self, **kwargs):
        pass
        '''for i in range(len(data.columns)-1):
            sns.jointplot(x=data.columns[i], y=data.columns[-1], data=data(), kind='reg')
            plt.show()'''

        '''for i in range(len(data.columns)-1):
            sns.jointplot(x=data.columns[i], y=data.columns[-1], data=data(), kind='kde')
            plt.show()'''

    def histplot(self, bins=40, **kwargs):
        self.hist(figsize=kwargs.get('figsize', (12, 12)), bins=bins)
        if kwargs.get('savefig', False): export2(plt, file_name=kwargs.get('title', 'histplot'), file_extension='png')

    def boxplot(self, scale=False, fill=True, **kwargs):
        plt.figure(figsize=kwargs.get('figsize', (12, 9)))
        plt.title(kwargs.get('title', 'boxplot'), fontsize=16, fontweight='bold')
        plt.grid(kwargs.get('grid', True))
        if not scale:
            sns.boxplot(self, fill=fill)
        else:
            sns.boxplot(pd.DataFrame(StandardScaler().fit_transform(self), columns=self.columns), fill=fill)
        if kwargs.get('savefig', False): export2(plt, file_name=kwargs.get('title', 'boxplot'), file_extension='png')

    def feature_target_split(self, **kwargs) -> tuple:
        """Разделение DataFrame на 2: features и target"""
        target = self.__get_target(**kwargs)
        return self.drop([target], axis=1), self[target]

    def __assert_split(self, **kwargs) -> None:
        """Проверка параметров разбиения"""

        random_state = kwargs.get('random_state', None)
        assert random_state is None or type(random_state) is int, 'random_state is None or type(random_state) is int'

    def train_test_split(self, test_size, shuffle: bool = True, **kwargs):
        """Разделение DataFrame на тренировочный и тестовый"""

        self.__assert_split(**kwargs)

        assert type(test_size) in (float, int), 'type(test_size) in (float, int)'

        assert type(shuffle) is bool, 'type(shuffle) is bool'

        # stratify не нужен в виду разбиение одного датафрейма self
        return train_test_split(self, test_size=test_size, shuffle=shuffle, **kwargs)

    def stratified_shuffle_split(self, n_splits: int, **kwargs):
        """"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        self.__assert_split(**kwargs)

        assert type(n_splits) is int, 'type(n_splits) is int'

        sss = StratifiedShuffleSplit(n_splits=n_splits, **kwargs)

        return sss.split(x, y, n_splits=n_splits, **kwargs)


if __name__ == '__main__':
    if 0:

        if 1:
            target = 'toxic'
            df.target = target

        if 0:
            print(DataFrame.vectorize_count)
            print(df.vectorize_count('comment'))
            print(df.vectorize_count(['comment']))
            print(df.vectorize_count(['comment'], stop_words=[]))
            print(df.vectorize_count(['comment'], stop_words=['00', 'ёмкость']))

            df.vectorize_count(['comment'], drop=True, inplace=True, stop_words=['00', 'ёмкость'])
            print(df)

        if 1:
            print(DataFrame.vectorize_tf_idf)
            print(df.vectorize_tf_idf('comment'))
            print(df.vectorize_tf_idf(['comment']))
            print(df.vectorize_tf_idf(['comment'], stop_words=[]))
            print(df.vectorize_tf_idf(['comment'], stop_words=['00', 'ёмкость']))

            df.vectorize_tf_idf(['comment'], drop=True, inplace=True, stop_words=['00', 'ёмкость'])
            print(df)

    if 1:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=True)
        df = pd.concat([data.data, data.target], axis=1)
        df = DataFrame(df)
        print(df)

        if 1:
            target = "MedHouseVal"
            df.target = target

        if 0:
            print(DataFrame.train_test_split.__name__)
            df.train_test_split(test_size=0.2)

        if 0:
            print(DataFrame.tsne)
            print(df.tsne(0))
            print(df.tsne(1))
            print(df.tsne(3))
            print(df.tsne(50))

        if 0:
            print(DataFrame.polynomial_features)
            print(df.polynomial_features(["Frequency [Hz]"], 3, True).columns)
            print('----------------')
            print(df.polynomial_features(["Frequency [Hz]", "Attack angle [deg]"], 4, True).columns)
            print('----------------')
            print(df.polynomial_features(["Frequency [Hz]"], 3, False).columns)
            print('----------------')
            print(df.polynomial_features(["Frequency [Hz]", "Attack angle [deg]"], 4, False).columns)
        if 0:
            print(DataFrame.detect_outliers.__name__)
            print(df.detect_outliers('Sigma'))
            print(df.detect_outliers('Tukey'))
            print(df.detect_outliers('Shovene'))

        if 0:
            print(DataFrame.corr_features.__name__)
            print(df.corr_features())
            print(df.corr_features(method='pearson'))
            print(df.corr_features(method='spearman'))
            print(df.corr_features(method='pearson', threshold=0.5))

        if 0:
            print(DataFrame.distribution.__name__)
            print(df.distribution([target]))
            print(df.distribution(df.columns.to_list()))

        if 0:
            print(DataFrame.variation_coefficient.__name__)
            print(df.variation_coefficient([target]))
            print(df.variation_coefficient(df.columns.to_list()))

        if 1:
            print(DataFrame.fill_na.__name__)

            print(df.fill_na(0))
            print(df.fill_na('0'))

            print(df.fill_na(method='mean'))
            print(df.fill_na(method='median'))
            print(df.fill_na(method='mode'))
            # print(df.fill_na(method='hmean'))
            print(df.fill_na(method='indicator'))

        if 1:
            print(DataFrame.l1_models.__name__)
            print(df.columns)
            l1 = list(2 ** np.linspace(-10, 0, 10))
            print(df.l1_importance(l1=l1, scale=True, early_stopping=True))
            print(df.l1_importance(l1=l1, scale=True, early_stopping=True, target=target))

        if 0:
            pass
            # print(df.find_corr_features())
            # print(df.encode_one_hot(["Frequency [Hz]"]))
            # print(df.mutual_info_score())
            # print(df.select_mutual_info_score_features(4))
            # print(df.select_mutual_info_score_features(2))
            # print(df.select_mutual_info_score_features(1.))
            # print(df.select_mutual_info_score_features(1.5))
            # print(df.select_mutual_info_score_features(-1.5))
            # print(df.mutual_info_score())
            # print(df.select_mutual_info_score_features(4))
            # print(df.select_mutual_info_score_features(1))
            # print(df.select_mutual_info_score_features(1.9))
            # print(df.select_mutual_info_score_features(50.))
            # print(df.importance_features())
            # print(df.select_importance_features(4))
            # print(df.select_importance_features(1))
            # print(df.select_importance_features(1.9))
            # print(df.select_importance_features(50.))

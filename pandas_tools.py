from tqdm import tqdm

import pandas as pd
import numpy as np

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
        """Получение target из словаря и приватного атрибута"""
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
        """Проверка на верность ввода в векторизатор"""

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

    # TODO
    def vectorize_tf_idf(self, columns: list[str], drop=False, inplace=False, **kwargs):
        """tf-idf векторизация токенов"""

        self.__assert_vectorize(**kwargs)

        corpus = self[columns].to_numpy().flatten()
        vectorizer = TfidfVectorizer(**kwargs)
        vectorizer.fit(self[columns].to_list())
        
        return DataFrame(vectorizer.transform(self[columns]).to_list())  # для преобразования из sparce matrix

    def detect_outliers(self, method: str = '3sigma'):
        """Обнаружение выбросов статистическим методом"""
        assert type(method) is str, f'{self.assert_sms} type(method) is str'
        method = method.strip().lower()
        assert method in ('3sigma', 'tukey'), f'{self.assert_sms} method in ("3sigma", "Tukey")!'

        outliers = DataFrame()
        for col in self.select_dtypes(include='number').columns:
            lower_bound, upper_bound = -np.inf, np.inf
            if method == '3sigma':  # если данные распределены нормально!
                mean = self[col].mean()
                std = self[col].std()
                lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
            elif method == 'tukey':
                q1, q3 = self[col].quantile(0.25), self[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            col_outliers = self[(self[col] < lower_bound) | (self[col] > upper_bound)]
            outliers = pd.concat([outliers, col_outliers])
        return DataFrame(outliers).sort_index().drop_duplicates()

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

    def select_corr_features(self, threshold: float = 0.85) -> list[str]:
        """Выбор линейно-независимых признаков"""
        com_m = self.corr().abs()  # матрица корреляции

        # верхний треугольник матрицы корреляции без диагонали
        uptriangle = com_m.where(np.triu(np.ones(com_m.shape), k=1).astype(bool))

        # индексы столбцов с корреляцией выше порога
        to_drop = [column for column in uptriangle.columns[1:]
                   if any(threshold > uptriangle[column]) or all(uptriangle[column].isna())]

        return to_drop

    def l1_models(self, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False,
                  **kwargs) -> list:
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

    def l1_importance(self, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False,
                      **kwargs):
        """Коэффициенты признаков линейной моедли с L1-регуляризацией"""
        target = self.__get_target(**kwargs)
        x, y = self.feature_target_split(target=target)

        l1_models = self.l1_models(l1=l1, scale=scale, early_stopping=early_stopping, target=target)
        df = DataFrame([l1_model.coef_ for l1_model in l1_models], columns=x.columns)
        return DataFrame(pd.concat([pd.DataFrame({'L1': l1}), df], axis=1))

    def l1_importance_plot(self, l1=tuple(2 ** np.linspace(-10, 10, 100)), scale=False, early_stopping=False,
                           **kwargs):
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

    def undersampling(self):
        """"""
        # Tomek Links
        pass

    def oversampling(self):
        """"""
        # SMOTE
        pass

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

    def tsnc(self, n_components: int, inplace=False, **kwargs):
        pass

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

    def train_test_split(self, test_size, shuffle=True, random_state=0):
        """Разделение DataFrame на тренировочный и тестовый"""
        # stratify не нужен в виду разбиение одного датафрейма self
        return train_test_split(self, test_size=test_size, shuffle=shuffle, random_state=random_state)


if __name__ == '__main__':
    if 1:
        df = DataFrame(pd.read_csv('russian_toxic_comments.csv'))
        print(df)

        if 1:
            print('vectorize_count')
            print(df.vectorize_count(['comment']))
            print(df.vectorize_count('comment'))
            print(df.vectorize_count(['comment'], stop_words=[]))
            print(df.vectorize_count(['comment'], stop_words=['00', 'ёмкость']))

            df.vectorize_count(['comment'], drop=True, inplace=True, stop_words=['00', 'ёмкость'])
            print(df)

    if 0:
        df = DataFrame(pd.read_csv('airfoil_self_noise.dat', sep="\t", header=None))
        df.columns = ["Frequency [Hz]", "Attack angle [deg]", "Chord length [m]", "Free-stream velocity [m/s]",
                      "Thickness [m]", "Pressure level [db]"]
        target = "Pressure level [db]"
        df.target = target
        print(df)
        print('----------------')
        print(df.polynomial_features(["Frequency [Hz]"], 3, True).columns)
        print('----------------')
        print(df.polynomial_features(["Frequency [Hz]", "Attack angle [deg]"], 4, True).columns)
        print('----------------')
        print(df.polynomial_features(["Frequency [Hz]"], 3, False).columns)
        print('----------------')
        print(df.polynomial_features(["Frequency [Hz]", "Attack angle [deg]"], 4, False).columns)

        # print(df.detect_outliers())
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

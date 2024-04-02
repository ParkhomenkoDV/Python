import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tools import export2


def detect_outliers(df, method='3sigma'):
    assert isinstance(df, pd.DataFrame), 'type(df) is DataFrame!'
    assert isinstance(method, str), 'type(method) is str!'
    method = method.strip().lower()
    assert method in ('3sigma', 'tukey'), 'method in ("3sigma", "Tukey")!'

    outliers = pd.DataFrame()
    for col in df.select_dtypes(include='number').columns:
        if method == '3sigma':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        elif method == 'tukey':
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, col_outliers])
    return outliers


def find_drop_features(com_m, threshold: float = 0.9):
    com_m = com_m.abs()

    # верхний треугольник матрицы корреляции без диагонали
    uptriangle = com_m.where(np.triu(np.ones(com_m.shape), k=1).astype(bool))

    # индексы столбцов с корреляцией выше порога
    to_drop = [column for column in uptriangle.columns[1:]
               if any(threshold <= uptriangle[column]) or all(uptriangle[column].isna())]

    return to_drop


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

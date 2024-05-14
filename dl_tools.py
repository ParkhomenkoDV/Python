import numpy as np
import matplotlib.pyplot as plt

from tools import export2


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


class NeuralNetwork:

    def predict(self):
        pass

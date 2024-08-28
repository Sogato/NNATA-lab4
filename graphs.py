import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")


def plot_roc_curve(roc_data, file_path='img/graphs_img/roc_curve.png'):
    """
    Строит и визуализирует ROC-кривые для каждой модели на основе рассчитанных данных AUC.
    Опционально сохраняет график в указанный файл.

    Аргументы:
        roc_data (dict): Словарь, где ключи - это имена моделей, а значения - словари с данными ROC-кривых
                         (fpr, tpr, roc_auc).
        file_path (str): Путь для сохранения графика. По умолчанию сохраняется в 'img/graphs_img/roc_curve.png'.
    """
    plt.figure(figsize=(12, 8))

    for model_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], lw=2, alpha=0.8,
                 label=f'{model_name} (AUC = {data["roc_auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=3, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Коэффициент ложных срабатываний', fontsize=14, fontweight='bold')
    plt.ylabel('Показатель истинных положительных результатов', fontsize=14, fontweight='bold')
    plt.title('ROC-кривые для моделей', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='-', alpha=0.7, linewidth=1.5)
    plt.legend(loc="lower right", fontsize=12)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'График сохранён в {file_path}')


def plot_training_history(history, model_name):
    """
    Строит графики изменения потерь и выбранной метрики на протяжении обучения модели.
    Графики отображают изменение значений на обучающем и валидационном наборах данных.

    Аргументы:
        history (History): Объект History от Keras, содержащий данные о процессе обучения модели.
        model_name (str): Название модели для отображения на графике и использования в имени файла.
    """
    plt.figure(figsize=(16, 8))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери на обучении', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Потери на валидации', color='orange', linewidth=2)
    plt.title(f'{model_name} - Потери', fontsize=16, fontweight='bold')
    plt.xlabel('Эпохи', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)

    # График метрик
    plt.subplot(1, 2, 2)
    metric = [key for key in history.history.keys() if 'accuracy' in key or 'auc' in key or 'rmse' in key]
    if metric:
        metric = metric[0]
        plt.plot(history.history[metric], label=f'Обучение {metric.upper()}', color='green', linewidth=2)
        plt.plot(history.history['val_' + metric], label=f'Валидация {metric.upper()}', color='red', linewidth=2)
        plt.title(f'{model_name} - {metric.upper()}', fontsize=16, fontweight='bold')
        plt.xlabel('Эпохи', fontsize=14)
        plt.ylabel(metric.upper(), fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True)

    plt.tight_layout()

    file_path = f'img/graphs_img/{model_name}_training_history.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'График сохранён в {file_path}')


def plot_test_metrics(metrics, model_name):
    """
    Строит горизонтальные столбчатые диаграммы для отображения метрик на тестовых данных.
    Отображает значения метрик на графике и сохраняет его в файл.

    Аргументы:
        metrics (dict): Словарь, где ключи - это названия метрик, а значения - их числовые значения.
        model_name (str): Название модели для отображения на графике и использования в имени файла.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(12, 8))
    bars = plt.barh(metric_names, metric_values, color=sns.color_palette("Blues_r", len(metric_values)))
    plt.title(f'{model_name} - Метрики на тестовых данных', fontsize=16, fontweight='bold')
    plt.xlabel('Значение', fontsize=14)

    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.4f}', va='center', fontsize=12)

    plt.tight_layout()

    file_path = f'img/graphs_img/{model_name}_test_metrics.png'
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f'График сохранён в {file_path}')


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve, auc, r2_score
from tensorflow.keras.callbacks import EarlyStopping

from graphs import plot_roc_curve, plot_training_history, plot_test_metrics


def load_data():
    """
    Загружает данные из CSV файлов.

    Возвращает:
        tuple: Три DataFrame, содержащие данные о диабете, физическом развитии и расходах домохозяйств.
    """
    diabetes_df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
    body_performance_df = pd.read_csv("data/bodyPerformance.csv")
    household_expenses_df = pd.read_csv("data/DS_2019_public.csv", low_memory=False)

    return diabetes_df, body_performance_df, household_expenses_df


def preprocess_diabetes_data(df):
    """
    Обрабатывает данные о диабете для задачи бинарной классификации.

    Аргументы:
        df (DataFrame): Исходный DataFrame с данными о диабете.

    Возвращает:
        tuple: Шесть объектов (X_train, X_val, X_test, y_train, y_val, y_test),
        представляющих обучающую, валидационную и тестовую выборки.
    """
    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_body_performance_data(df):
    """
    Обрабатывает данные о физическом развитии для задачи многоклассовой классификации.

    Аргументы:
        df (DataFrame): Исходный DataFrame с данными о физическом развитии.

    Возвращает:
        tuple: Шесть объектов (X_train, X_val, X_test, y_train, y_val, y_test),
        представляющих обучающую, валидационную и тестовую выборки.
    """
    df['gender'] = LabelEncoder().fit_transform(df['gender'])

    X = df.drop(columns=['class'])
    y = df['class']
    y = LabelEncoder().fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_household_expenses_data(df, target_column):
    """
    Обрабатывает данные о расходах домохозяйств для задачи регрессии.

    Аргументы:
        df (DataFrame): Исходный DataFrame с данными о расходах домохозяйств.
        target_column (str): Название столбца, содержащего целевую переменную.

    Возвращает:
        tuple: Шесть объектов (X_train, X_val, X_test, y_train, y_val, y_test),
        представляющих обучающую, валидационную и тестовую выборки.
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_binary_classifier(input_shape):
    """
    Создает и компилирует модель бинарного классификатора на основе полносвязных слоев.

    Аргументы:
        input_shape (int): Размерность входных данных.

    Возвращает:
        model (Sequential): Компилированная модель бинарного классификатора.
    """
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model


def build_multiclass_classifier(input_shape, num_classes):
    """
    Создает и компилирует модель многоклассового классификатора на основе полносвязных слоев.

    Аргументы:
        input_shape (int): Размерность входных данных.
        num_classes (int): Количество классов для классификации.

    Возвращает:
        model (Sequential): Компилированная модель многоклассового классификатора.
    """
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_regressor(input_shape):
    """
    Создает и компилирует модель регрессора на основе полносвязных слоев.

    Аргументы:
        input_shape (int): Размерность входных данных.

    Возвращает:
        model (Sequential): Компилированная модель регрессора.
    """
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error',
                           tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    return model


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, task_type="classification",
                             num_classes=None, epochs=500, batch_size=32):
    """
    Обучает и оценивает модель на данных, выводит результаты и метрики.

    Аргументы:
        model (Sequential): Модель для обучения и оценки.
        X_train (array-like): Обучающая выборка признаков.
        y_train (array-like): Обучающая выборка меток/целевых переменных.
        X_val (array-like): Валидационная выборка признаков.
        y_val (array-like): Валидационная выборка меток/целевых переменных.
        X_test (array-like): Тестовая выборка признаков.
        y_test (array-like): Тестовая выборка меток/целевых переменных.
        task_type (str): Тип задачи ("binary_classification", "multiclass_classification", "regression").
        num_classes (int, optional): Количество классов (для многоклассовой классификации).
        epochs (int, optional): Количество эпох для обучения. По умолчанию 500.
        batch_size (int, optional): Размер батча для обучения. По умолчанию 32.

    Возвращает:
        tuple: История обучения (history), метрики тестирования (test_metrics) и данные ROC (roc_data).
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        callbacks=[early_stopping])

    test_metrics = model.evaluate(X_test, y_test, verbose=2)

    roc_data = {}

    if task_type == "binary_classification":
        y_pred = model.predict(X_test).ravel()
        y_pred_binary = (y_pred > 0.5).astype("int32")

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        roc_data["Binary Classifier"] = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
        print("Метрики Binary Classifier:")
        print("Recall:", recall_score(y_test, y_pred_binary))
        print("Precision:", precision_score(y_test, y_pred_binary))
        print("Weighted Accuracy:", accuracy_score(y_test, y_pred_binary))
        print("AUC:", roc_auc)

    elif task_type == "multiclass_classification":
        y_pred = model.predict(X_test).argmax(axis=1)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test, model.predict(X_test)[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_data[f'Class {i}'] = {"fpr": fpr[i], "tpr": tpr[i], "roc_auc": roc_auc[i]}

        print("Метрики Multiclass Classifier:")
        print("Recall:", recall_score(y_test, y_pred, average="weighted"))
        print("Precision:", precision_score(y_test, y_pred, average="weighted"))
        print("Weighted Accuracy:", accuracy_score(y_test, y_pred))
        print("AUC:", roc_auc_score(y_test, model.predict(X_test), multi_class='ovr'))

    elif task_type == "regression":
        y_pred = model.predict(X_test)

        print("Метрики Regressor:")
        print("MSE:", test_metrics[1])
        print("MAE:", test_metrics[0])
        print("R2:", r2_score(y_test, y_pred))

    return history, test_metrics, roc_data


if __name__ == "__main__":
    diabetes_df, body_performance_df, household_expenses_df = load_data()

    X_train_diabetes, X_val_diabetes, X_test_diabetes, y_train_diabetes, y_val_diabetes, y_test_diabetes = preprocess_diabetes_data(diabetes_df)
    X_train_body, X_val_body, X_test_body, y_train_body, y_val_body, y_test_body = preprocess_body_performance_data(body_performance_df)
    X_train_household, X_val_household, X_test_household, y_train_household, y_val_household, y_test_household = preprocess_household_expenses_data(household_expenses_df, target_column='TOTALDOL')

    binary_classifier = build_binary_classifier(X_train_diabetes.shape[1])
    binary_history, binary_test_metrics, binary_roc_data = train_and_evaluate_model(binary_classifier,
                                                                   X_train_diabetes, y_train_diabetes,
                                                                   X_val_diabetes, y_val_diabetes,
                                                                   X_test_diabetes, y_test_diabetes,
                                                                   task_type="binary_classification")
    binary_metrics = {
        "Recall": recall_score(y_test_diabetes, (binary_classifier.predict(X_test_diabetes) > 0.5).astype("int32")),
        "Precision": precision_score(y_test_diabetes, (binary_classifier.predict(X_test_diabetes) > 0.5).astype("int32")),
        "Accuracy": accuracy_score(y_test_diabetes, (binary_classifier.predict(X_test_diabetes) > 0.5).astype("int32")),
        "AUC": binary_test_metrics[1]
    }
    plot_training_history(binary_history, model_name="Binary Classifier")
    plot_test_metrics(binary_metrics, model_name="Binary Classifier")

    multiclass_classifier = build_multiclass_classifier(X_train_body.shape[1], num_classes=4)
    multiclass_history, multiclass_test_metrics, multiclass_roc_data = train_and_evaluate_model(multiclass_classifier,
                                                                           X_train_body, y_train_body,
                                                                           X_val_body, y_val_body,
                                                                           X_test_body, y_test_body,
                                                                           task_type="multiclass_classification",
                                                                           num_classes=4)
    multiclass_metrics = {
        "Recall": recall_score(y_test_body, multiclass_classifier.predict(X_test_body).argmax(axis=1), average="weighted"),
        "Precision": precision_score(y_test_body, multiclass_classifier.predict(X_test_body).argmax(axis=1), average="weighted"),
        "Accuracy": accuracy_score(y_test_body, multiclass_classifier.predict(X_test_body).argmax(axis=1)),
        "AUC": roc_auc_score(y_test_body, multiclass_classifier.predict(X_test_body), multi_class='ovr')
    }
    plot_training_history(multiclass_history, model_name="Multiclass Classifier")
    plot_test_metrics(multiclass_metrics, model_name="Multiclass Classifier")

    regressor = build_regressor(X_train_household.shape[1])
    regressor_history, regressor_test_metrics, _ = train_and_evaluate_model(regressor,
                                                                         X_train_household, y_train_household,
                                                                         X_val_household, y_val_household,
                                                                         X_test_household, y_test_household,
                                                                         task_type="regression")
    regressor_metrics = {
        "MSE": regressor_test_metrics[1],
        "MAE": regressor_test_metrics[0],
        "R2": r2_score(y_test_household, regressor.predict(X_test_household))
    }
    plot_training_history(regressor_history, model_name="Regressor")
    plot_test_metrics(regressor_metrics, model_name="Regressor")
    plot_roc_curve({**binary_roc_data, **multiclass_roc_data})

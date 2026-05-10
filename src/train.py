"""Модуль обучения модели с сохранением экспериментов."""
import configparser
import hashlib
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from logger import Logger

SHOW_LOG = True

class ModelTrainer:
    """Класс обучения RandomForest и управления экспериментами."""

    def __init__(self) -> None:
        """Инициализация тренера модели с конфигурацией."""
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.exp_root = self.config["PATHS"]["experiments"]
        os.makedirs(self.exp_root, exist_ok=True)
        self.log.info("ModelTrainer инициализирован")

    async def train(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Обучение модели RandomForestClassifier и фиксация эксперимента.

        Args:
            X_train: обучающие признаки.
            X_test: тестовые признаки.
            y_train: обучающие метки.
            y_test: тестовые метки.

        Returns:
            dict: результат обучения с ключами status, f1_score, experiment_path.
        """
        # Данные приходят уже отмасштабированными из DataPreprocessor
        params = {
            "n_estimators": self.config.getint("MODEL", "n_estimators"),
            "criterion": self.config["MODEL"]["criterion"],
            "max_depth": self.config.getint("MODEL", "max_depth"),
            "random_state": self.config.getint("MODEL", "random_state"),
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "f1_score": float(f1_score(y_test, preds, average="weighted")),
            "accuracy": float(accuracy_score(y_test, preds)),
        }
        self.log.info(f"Обучение завершено: {metrics}")

        exp_dir = self._save_experiment(model, None, params, metrics)
        return {
            "status": "success",
            "f1_score": metrics["f1_score"],
            "experiment_path": exp_dir
        }

    def _save_experiment(self, model, scaler, params, metrics) -> str:
        """
        Сохранение артефактов эксперимента в подпапку.

        Args:
            model: обученная модель.
            scaler: StandardScaler (не используется, т.к. масштабирование в препроцессоре).
            params: гиперпараметры модели.
            metrics: словарь с метриками.

        Returns:
            str: путь к папке эксперимента.
        """
        exp_id = len(os.listdir(self.exp_root)) + 1
        exp_dir = os.path.join(self.exp_root, f"exp_{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)

        # Сохранение модели
        with open(os.path.join(exp_dir, "trained_model.pkl"), "wb") as f:
            pickle.dump(model, f)

        # Хэш модели
        with open(os.path.join(exp_dir, "trained_model.pkl"), "rb") as f:
            model_hash = hashlib.md5(f.read()).hexdigest()

        # config.yml
        config_data = {
            "model_type": "RandomForestClassifier",
            "params": params,
            "data_source": "postgresql:glass",
            "log_dir": "logfile.log",
            "model_hash": model_hash,
        }
        with open(os.path.join(exp_dir, "config.yml"), "w") as f:
            yaml.safe_dump(config_data, f)

        # metrics.yml
        with open(os.path.join(exp_dir, "metrics.yml"), "w") as f:
            yaml.safe_dump(metrics, f)

        # logs.txt
        if os.path.exists("logfile.log"):
            with open("logfile.log", "r") as src, \
                 open(os.path.join(exp_dir, "logs.txt"), "w") as dst:
                dst.write(src.read())

        self.log.info(f"Эксперимент сохранён: {exp_dir}")
        return exp_dir
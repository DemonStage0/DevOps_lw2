"""FastAPI-сервис для классификации типов стекла."""
import sys
import os
from fastapi import FastAPI, HTTPException, Query

# Добавляем src в путь для импортов
sys.path.insert(0, os.path.dirname(__file__))

from logger import Logger
from database import get_glass_data, save_prediction
from preprocess import DataPreprocessor
from train import ModelTrainer
from predict import Predictor

app = FastAPI(title="Glass Classification API", version="2.0.0")
log = Logger(True).get_logger(__name__)


@app.get("/train")
async def train():
    """Обучает модель на данных из БД и сохраняет эксперимент."""
    X, y = await get_glass_data()
    preprocessor = DataPreprocessor()
    X_tr, X_te, y_tr, y_te = await preprocessor.prepare_data(X, y)
    trainer = ModelTrainer()
    result = await trainer.train(X_tr, X_te, y_tr, y_te)
    return {"message": f"Модель обучена успешно. F1 = {result['f1_score']:.4f}"}


@app.get("/predict")
async def predict(
    RI: float = Query(1.52101), Na: float = Query(13.64), Mg: float = Query(4.49),
    Al: float = Query(1.1), Si: float = Query(71.78), K: float = Query(0.06),
    Ca: float = Query(8.75), Ba: float = Query(0.0), Fe: float = Query(0.0),
):
    """Предсказывает класс стекла и сохраняет результат в БД."""
    features = [RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]
    try:
        pred = Predictor().predict(features)
    except FileNotFoundError:
        raise HTTPException(404, "Нет предобученной модели. Сначала вызовите /train")
    await save_prediction(pred, features)
    return {"predicted_class": pred}
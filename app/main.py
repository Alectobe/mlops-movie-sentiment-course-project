from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import yaml

# ---------- Конфигурация путей ----------
# Предполагаем, что папка app/ лежит в корне проекта рядом с models/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    full_config = yaml.safe_load(f)

inference_cfg = full_config["inference"]

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(BASE_DIR, inference_cfg["model_path"])
VECTORIZER_PATH = os.path.join(BASE_DIR, inference_cfg["vectorizer_path"])

# ---------- Загрузка модели и векторизатора при старте сервиса ----------
# Делаем это один раз, чтобы не грузить модель при каждом запросе

logreg_model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

# ---------- FastAPI-приложение ----------
app = FastAPI(
    title="IMDB Sentiment Analysis Service",
    description="Сервис для предсказания тональности отзывов о фильмах (positive / negative)",
    version="1.0.0",
)


# ---------- Pydantic-схемы ----------
class PredictRequest(BaseModel):
    """Схема входных данных для эндпоинта /predict."""
    review: str


class PredictResponse(BaseModel):
    """Схема выходных данных /predict."""
    sentiment: str
    positive_proba: float
    negative_proba: float


# ---------- Эндпоинты ----------

@app.get("/health")
def healthcheck():
    """
    Простой эндпоинт для проверки, что сервис жив.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Принимает текст отзыва и возвращает предсказанную тональность.
    """
    # Получаем текст из запроса
    text = request.review

    # Преобразуем в список, потому что vectorizer ждёт итерабельный объект
    texts = [text]

    # Векторизация
    X_vec = tfidf_vectorizer.transform(texts)

    # Предсказание класса
    pred_label = logreg_model.predict(X_vec)[0]

    # Предсказание вероятностей
    proba = logreg_model.predict_proba(X_vec)[0]
    # В sklearn порядок соответствует model.classes_
    classes = list(logreg_model.classes_)
    positive_idx = classes.index("positive")
    negative_idx = classes.index("negative")

    positive_proba = float(proba[positive_idx])
    negative_proba = float(proba[negative_idx])

    return PredictResponse(
        sentiment=pred_label,
        positive_proba=round(positive_proba, 4),
        negative_proba=round(negative_proba, 4),
    )
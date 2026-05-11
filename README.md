# DevOps_lw2 - Glass Classification ML Pipeline с PostgreSQL

## Описание проекта
Проект реализует классический жизненный цикл разработки ML модели для классификации типов стекла с использованием CI/CD пайплайна. Модель классифицирует стекло по 7 классам на основе 9 химических признаков. Данные хранятся в базе данных PostgreSQL, а сервис модели реализован с использованием FastAPI и SQLAlchemy.

## Структура проекта
```
DevOps_lw2/
├── CI/
│   └── Jenkinsfile          # CI пайплайн
├── CD/
│   └── Jenkinsfile          # CD пайплайн
├── data/
│   ├── glass.csv            # Исходный датасет (для первоначальной загрузки в БД)
│   └── init.sql             # SQL-скрипт создания таблиц
├── src/
│   ├── unit_tests/
│   │   ├── test_api.py          # Тесты API эндпоинтов
│   │   ├── test_predict.py      # Тесты предиктора
│   │   ├── test_preprocess.py   # Тесты предобработки
│   │   └── test_training.py     # Тесты обучения
│   ├── app.py               # FastAPI приложение
│   ├── csv_to_db.py         # Скрипт переноса данных из CSV в PostgreSQL
│   ├── database.py          # Модели SQLAlchemy и подключение к БД
│   ├── logger.py            # Модуль логирования
│   ├── predict.py           # Класс для предсказаний
│   ├── preprocess.py        # Предобработка данных
│   └── train.py             # Обучение модели
├── tests/
│   ├── test_0.json          # Тестовые данные для функционального тестирования
│   └── test_1.json          # Тестовые данные для функционального тестирования
├── .env                     # Переменные окружения для подключения к БД (в .gitignore)
├── .gitignore
├── config.ini               # Гиперпараметры модели
├── docker-compose.yml       # Конфигурация сервисов (БД + приложение)
├── Dockerfile               # Сборка Docker образа
├── README.md
├── requirements.txt         # Зависимости проекта
└── requirements_freeze.txt  # Зафиксированные версии зависимостей
```

## Установка и запуск

### Предварительные требования
- Docker и Docker Compose
- Git

### Быстрый запуск (рекомендуется)
```bash
# Клонирование репозитория
git clone https://github.com/DemonStage0/DevOps_lw2.git
cd DevOps_lw2

# Создание .env файла с параметрами подключения к БД
echo DB_HOST=db > .env
echo DB_PORT=5432 >> .env
echo DB_USER=postgres >> .env
echo DB_PASS=postgres >> .env
echo DB_NAME=glass_db >> .env

# Запуск всех сервисов (PostgreSQL + инициализация БД + FastAPI)
docker-compose up -d --build
```

### Порядок запуска сервисов
1. **PostgreSQL** — контейнер `glass_db` с автоматическим созданием таблиц через `init.sql`
2. **Инициализация БД** — контейнер `glass_init` переносит данные из `glass.csv` в таблицу `glass`
3. **FastAPI** — контейнер `devops_lw2-web-1` запускает API на порту 8000 после готовности БД

### Локальный запуск (для разработки)
```bash
pip install -r requirements.txt
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health check
```
GET /
Ответ: {"message": "Glass Classification API is running", "version": "2.0.0"}
```

### Обучение модели
```
GET /train
Ответ: {"message": "Модель обучена успешно. F1 = 0.7542"}
Примечание: данные для обучения загружаются из таблицы glass в PostgreSQL
```

### Предсказание класса стекла
```
GET /predict?RI=1.52101&Na=13.64&Mg=4.49&Al=1.1&Si=71.78&K=0.06&Ca=8.75&Ba=0.0&Fe=0.0
Ответ: {"predicted_class": 1}
```
Все 9 параметров обязательны, имеют значения по умолчанию.

### Классы стекла
```
1 - building_windows_float_processed
2 - building_windows_non_float_processed
3 - vehicle_windows_float_processed
4 - vehicle_windows_non_float_processed (отсутствует в датасете)
5 - containers
6 - tableware
7 - headlamps
```

## База данных

### Структура таблиц
**glass** — обучающие данные (заполняется из `glass.csv`):
- id, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Type

**predict** — результаты предсказаний:
- id, predicted_class, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, timestamp

### Подключение
Параметры подключения задаются в `.env` файле:
```
DB_HOST=db
DB_PORT=5432
DB_USER=postgres
DB_PASS=postgres
DB_NAME=glass_db
```

## Тестирование

### Unit-тесты
```bash
python -m pytest src/unit_tests/ -v
```

### Функциональные тесты API
```bash
curl http://localhost:8000/
curl http://localhost:8000/train
curl "http://localhost:8000/predict?RI=1.52101&Na=13.64&Mg=4.49&Al=1.1&Si=71.78&K=0.06&Ca=8.75&Ba=0.0&Fe=0.0"
```

## CI/CD Pipeline

### CI Pipeline (Jenkins)
- Клонирование репозитория из GitHub
- Сборка Docker образов для сервисов
- Запуск контейнеров (PostgreSQL + FastAPI)
- Запуск unit-тестов внутри контейнера
- Функциональное тестирование API эндпоинтов
- Публикация образа в Docker Hub

### CD Pipeline (Jenkins)
- Загрузка образов из Docker Hub
- Запуск полного стека (БД + приложение)
- Функциональное тестирование эндпоинтов
- Проверка корректности предсказаний

## Эксперименты
Каждый эксперимент сохраняется в `experiments/exp_N/` и содержит:
- `config.yml` — параметры модели, хэш обученной модели
- `trained_model.pkl` — сериализованная модель RandomForestClassifier
- `metrics.yml` — метрики качества (F1, accuracy)
- `logs.txt` — логи обучения

## Технологии
- **Python 3.12** + scikit-learn (Random Forest Classifier)
- **FastAPI** + Uvicorn (REST API)
- **PostgreSQL 15** (хранение данных и результатов)
- **SQLAlchemy 2.0** + asyncpg (асинхронная работа с БД)
- **Pydantic Settings** (управление конфигурацией)
- **Docker** + Docker Compose (контейнеризация)
- **Jenkins** (CI/CD пайплайны)
- **Git** + GitHub (контроль версий)
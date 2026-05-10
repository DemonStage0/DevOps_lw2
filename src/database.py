"""Модуль для взаимодействия с базой данных через SQLAlchemy."""
from datetime import datetime
from typing import List
from sqlalchemy import Column, Integer, Float, DateTime, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Настройки подключения к БД, загружаемые из .env."""
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: str
    DB_NAME: str

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        env_file = ".env"

settings = Settings()

engine = create_async_engine(settings.DATABASE_URL)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    """Базовый класс для ORM-моделей."""
    pass

class Glass(Base):
    """Модель таблицы glass (обучающие данные)."""
    __tablename__ = "glass"
    id = Column(Integer, primary_key=True, index=True)
    RI = Column("RI", Float, nullable=False)
    Na = Column("Na", Float, nullable=False)
    Mg = Column("Mg", Float, nullable=False)
    Al = Column("Al", Float, nullable=False)
    Si = Column("Si", Float, nullable=False)
    K = Column("K", Float, nullable=False)
    Ca = Column("Ca", Float, nullable=False)
    Ba = Column("Ba", Float, nullable=False)
    Fe = Column("Fe", Float, nullable=False)
    Type = Column("Type", Integer, nullable=False)

class Predict(Base):
    """Модель таблицы predict (результаты предсказаний)."""
    __tablename__ = "predict"
    id = Column(Integer, primary_key=True, index=True)
    predicted_class = Column(Integer, nullable=False)
    RI = Column("RI", Float, nullable=False)
    Na = Column("Na", Float, nullable=False)
    Mg = Column("Mg", Float, nullable=False)
    Al = Column("Al", Float, nullable=False)
    Si = Column("Si", Float, nullable=False)
    K = Column("K", Float, nullable=False)
    Ca = Column("Ca", Float, nullable=False)
    Ba = Column("Ba", Float, nullable=False)
    Fe = Column("Fe", Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

async def get_glass_data() -> tuple:
    """Получение всех записей из таблицы glass в виде X (признаки) и y (метки)."""
    async with async_session() as session:
        result = await session.execute(select(Glass))
        rows = result.scalars().all()
        X = [[r.RI, r.Na, r.Mg, r.Al, r.Si, r.K, r.Ca, r.Ba, r.Fe] for r in rows]
        y = [r.Type for r in rows]
        return X, y

async def save_prediction(predicted_class: int, features: List[float]) -> None:
    """Сохранение результата предсказания в таблицу predict."""
    async with async_session() as session:
        record = Predict(
            predicted_class=predicted_class,
            RI=features[0],
            Na=features[1],
            Mg=features[2],
            Al=features[3],
            Si=features[4],
            K=features[5],
            Ca=features[6],
            Ba=features[7],
            Fe=features[8],
        )
        session.add(record)
        await session.commit()
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector


# ──────────────────────────────────────────────
# Подключение к БД
# ──────────────────────────────────────────────

DATABASE_URL = "postgresql://user:password@localhost:5434/dbname"

engine = create_engine(DATABASE_URL, echo=False)


# ──────────────────────────────────────────────
# Базовый класс моделей
# ──────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────
# Модель таблицы attractions_embedding
# ──────────────────────────────────────────────

class AttractionEmbedding(Base):
    __tablename__ = "attractions_embedding"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    attraction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    vector: Mapped[list[float]] = mapped_column(Vector(512))
    computed_at: Mapped[datetime] = mapped_column()
# ──────────────────────────────────────────────
# Метод сохранения вектора в БД
# ──────────────────────────────────────────────

def save_attraction_embedding(
    attraction_id: uuid.UUID,
    vector: list[float],
) -> AttractionEmbedding:
    """
    Сохраняет векторное представление достопримечательности в БД.
    Если запись для данной достопримечательности уже существует —
    обновляет вектор и время вычисления.

    Параметры:
        attraction_id — UUID достопримечательности
        vector        — список из 512 float-значений

    Возвращает:
        Сохранённый объект AttractionEmbedding
    """
    if len(vector) != 512:
        raise ValueError(f"Ожидается вектор размерности 512, получено {len(vector)}")

    with Session(engine) as session:
        # Проверяем, есть ли уже запись для этой достопримечательности
        existing: AttractionEmbedding | None = (
            session.query(AttractionEmbedding)
            .filter_by(attraction_id=attraction_id)
            .first()
        )

        if existing:
            # Обновляем существующую запись
            existing.vector = vector
            existing.computed_at = datetime.now(timezone.utc)
            record = existing
        else:
            # Создаём новую запись
            record = AttractionEmbedding(
                attraction_id=attraction_id,
                vector=vector,
            )
            session.add(record)

        session.commit()
        session.refresh(record)

    return record
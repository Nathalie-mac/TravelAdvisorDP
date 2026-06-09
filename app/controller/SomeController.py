from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.recommendations import get_similar_attractions

router = APIRouter(prefix="/api/v1", tags=["recommend"])


class RecommendRequest(BaseModel):
    current_attraction_id: int
    profile_id: int


class RecommendResponse(BaseModel):
    attraction_ids: list[int]


@router.get("/recommend", response_model=RecommendResponse)
async def recommend_similar(request: RecommendRequest) -> RecommendResponse:
    """
    Возвращает топ-10 похожих достопримечательностей для блока
    «Похожие места» на карточке объекта.

    Тело запроса:
        current_attraction_id — id просматриваемой достопримечательности
        profile_id            — id профиля пользователя

    Ответ:
        attraction_ids — массив из 10 id достопримечательностей,
                         отсортированных по убыванию релевантности
    """
    attraction_ids = get_similar_attractions(
        current_attraction_id=request.current_attraction_id,
        profile_id=request.profile_id,
    )

    if not attraction_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Эмбеддинг для объекта {request.current_attraction_id} не найден",
        )

    return RecommendResponse(attraction_ids=attraction_ids)
import numpy as np
from typing import Optional


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Косинусное сходство двух L2-нормализованных векторов."""
    return float(np.dot(vec_a, vec_b))


# ──────────────────────────────────────────────
# Алгоритм 1: «Для вас»
# ──────────────────────────────────────────────

def get_recommendations_for_you(profile_id: int) -> list[int]:
    """
    Подбирает топ-10 рекомендаций для страницы «Для вас».

    Параметры:
        profile_id — идентификатор профиля пользователя

    Возвращает:
        Список из 10 attraction_id, отсортированных по убыванию
        косинусного сходства с вектором профиля пользователя.
    """

    # Получаем вектор профиля пользователя из БД
    profile_vector: np.ndarray = get_profile_vector_from_db(profile_id)

    # Получаем массив id достопримечательностей из категорий,
    # выбранных пользователем на онбординге
    attraction_ids: list[int] = get_attraction_ids_by_onboarding_categories(profile_id)

    # Получаем массив id объектов, с которыми пользователь уже взаимодействовал
    attraction_interacted_ids: list[int] = get_interacted_attraction_ids(profile_id)

    # Фильтрация: если есть история взаимодействий — исключаем просмотренные
    if attraction_interacted_ids:
        interacted_set = set(attraction_interacted_ids)
        attraction_filtered_ids = [
            aid for aid in attraction_ids
            if aid not in interacted_set
        ]
    else:
        # Истории нет — работаем со всем массивом без фильтрации
        attraction_filtered_ids = attraction_ids

    # Вычисляем косинусное сходство профиля с каждым объектом
    scored_ids: dict[int, float] = {}

    for attraction_id in attraction_filtered_ids:
        attraction_vector: np.ndarray = get_attraction_vector_from_db(attraction_id)
        score = cosine_similarity(profile_vector, attraction_vector)
        scored_ids[attraction_id] = score

    # Сортируем по убыванию score
    scored_ids = dict(
        sorted(scored_ids.items(), key=lambda item: item[1], reverse=True)
    )

    # Берём топ-10
    top_attraction_ids = list(scored_ids.keys())[:10]

    # Возвращаем массив id микросервису достопримечательностей
    return top_attraction_ids


# ──────────────────────────────────────────────
# Алгоритм 2: «Похожие места» на карточке объекта
# ──────────────────────────────────────────────

def get_similar_attractions(current_attraction_id: int,
                             profile_id: int,
                            alpha: float = 0.7) -> list[int]:
    """
    Подбирает топ-10 похожих достопримечательностей для блока
    под карточкой открытого объекта.

    Параметры:
        current_attraction_id — id просматриваемой достопримечательности
        profile_id            — id профиля пользователя
        alpha                 — вес сходства с текущим объектом (0..1),
                                (1 - alpha) — вес сходства с профилем пользователя
    Возвращает:
        Список из 10 attraction_id, отсортированных по убыванию
        косинусного сходства с вектором текущего объекта,
        или пустой список, если эмбеддинг объекта не найден.
    """

    # Получаем эмбеддинг просматриваемого объекта
    current_vector: Optional[np.ndarray] = get_attraction_vector_from_db(
        current_attraction_id
    )

    # Если эмбеддинг не найден — возвращаем пустой массив
    if current_vector is None:
        return []

    # Получаем вектор профиля пользователя
    profile_vector: np.ndarray = get_profile_vector_from_db(profile_id)

    # Получаем словарь всех объектов {attraction_id: attraction_vector},
    # исключая текущий объект
    attractions: dict[int, np.ndarray] = get_all_attraction_vectors_except(
        current_attraction_id
    )

    attraction_ids: list[int] = []

    # Вычисляем косинусное сходство текущего объекта с каждым остальным
    scored_ids: dict[int, float] = {}

    for attraction_id, attraction_vector in attractions.items():
        sim_object = cosine_similarity(current_vector, attraction_vector)
        sim_profile = cosine_similarity(profile_vector, attraction_vector)
        score = alpha * sim_object + (1 - alpha) * sim_profile
        scored_ids[attraction_id] = score

    # Сортируем по убыванию score
    scored_ids = dict(
        sorted(scored_ids.items(), key=lambda item: item[1], reverse=True)
    )

    # Берём топ-10, записываем в attraction_ids
    attraction_ids = list(scored_ids.keys())[:10]

    # Возвращаем массив id микросервису достопримечательностей
    return attraction_ids


# ──────────────────────────────────────────────
# Заглушки методов работы с БД и микросервисом
# (логика инкапсулирована, здесь не реализуется)
# ──────────────────────────────────────────────

def get_profile_vector_from_db(profile_id: int) -> np.ndarray: ...
def get_attraction_vector_from_db(attraction_id: int) -> Optional[np.ndarray]: ...
def get_attraction_ids_by_onboarding_categories(profile_id: int) -> list[int]: ...
def get_interacted_attraction_ids(profile_id: int) -> list[int]: ...
def get_all_attraction_vectors_except(exclude_id: int) -> dict[int, np.ndarray]: ...
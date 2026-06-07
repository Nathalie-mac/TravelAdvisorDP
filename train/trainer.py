from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

model_id = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
local_dir = os.path.abspath("/home/nathaliemac/PycharmProjects/TravelAdvisorDP/data/models/local_model")

model = SentenceTransformer(local_dir)
#model.save(local_dir)
attractions = [
    "Третьяковская галерея, Москва. Крупнейший музей русского искусства, коллекция икон и картин от древнерусской живописи до авангарда XX века, аудиогиды и обзорные экскурсии.",
    "Фестиваль «Спасская башня», Москва. Ежегодный международный военный музыкальный фестиваль на Красной площади, показательные выступления духовых оркестров, световое шоу и парад конницы.",
    "Озеро Байкал, Иркутская область. Самое глубокое озеро в мире, кристально чистая вода, эндемичная фауна, пешие тропы вдоль берега, наблюдение за закатами и экологические маршруты.",
    "Тропа «Экология», Красная Поляна. Маршрут средней сложности через альпийские луга и хвойные леса Кавказского заповедника, оборудованные тропы, смотровые площадки, перепады высот до 800 м.",
    "Горнолыжный курорт «Роза Хутор», Сочи. Олимпийский курорт с трассами разного уровня сложности, школа инструкторов, прокат снаряжения, соревнования по слалому и сноуборду, инфраструктура для активных тренировок.",
    "Парашютный клуб «Аэроград Коломна», Московская область. Прыжки с парашютом в тандеме и самостоятельно, аэротруба для тренировок, акробатические полёты, сертификация инструкторов и адреналиновые программы для новичков.",
    "Гастрономический маршрут «Казанские традиции», Казань. Дегустации национальной татарской кухни, мастер-классы по выпечке эчпочмаков, посещение фермерских рынков, история специй и локальных продуктов в сопровождении гидов.",
    "Пляж «Лазурный берег», Анапа. Песчаная коса с пологим входом в море, шезлонги, пляжные кафе, водные развлечения, детские площадки и вечерние развлекательные программы у воды.",
    "Санаторий «Матрешка Резорт», Кисловодск. Лечебные нарзанные ванны, климатотерапия в курортном парке, SPA-процедуры, массаж и йога, программы реабилитации и профилактики заболеваний опорно-двигательного аппарата."
]

names = [
    "Третьяковская галерея (Культурно-познавательный)",
    "Фестиваль «Спасская башня» (Событийный)",
    "Озеро Байкал (Природный)",
    "Тропа «Экология» (Хайкинг)",
    "Курорт «Роза Хутор» (Спортивный)",
    "Парашютный клуб «Аэроград» (Экстремальный)",
    "Маршрут «Казанские традиции» (Гастрономический)",
    "Пляж «Лазурный берег» (Пляжный)",
    "Санаторий «Матрешка Резорт» (Оздоровительный)"
]
embeddings = model.encode(attractions, normalize_embeddings=True)

print("Размерность вектора:", embeddings.shape)
print("Пример первых 5 значений 1-го вектора:", embeddings[0][:5])

#similarity = model.similarity(embeddings[0], embeddings[1])
similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()

# 5. Визуализация через pandas (тепловая таблица)
df_sim = pd.DataFrame(similarity_matrix, index=names, columns=names)
#print(f"Сходство 'Музей' ↔ 'Пляж': {similarity.item():.4f}")


# print("=" * 120)
# # Округляем до 2 знаков для читаемости
# print(df_sim.round(2).to_string())
# print("=" * 120)

plt.figure(figsize=(12, 10))

white_cmap = ListedColormap(['white'])
# Создаём heatmap с монохромным стилем
ax = sns.heatmap(
    df_sim.round(2),
    annot=True,                 # Показываем значения в ячейках
    fmt=".2f",                  # Формат чисел
    linewidths=1,               # Более заметные линии сетки
    linecolor="black",          # Чёрные границы ячеек
    square=True,                # Квадратные ячейки
    cmap=white_cmap,
    cbar=False,                 # Убираем цветовую легенду справа
    annot_kws={"size": 9, "weight": "bold", "color": "black"}  # Чёрный текст
)

# Переносим подписи оси X наверх
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Настройка заголовка и подписей
plt.title(
    "Матрица семантического сходства достопримечательностей",
    fontsize=14,
    fontweight="bold",
    pad=20
)
plt.xticks(fontsize=9, rotation=45, ha="left")   # Подписи сверху, поворот влево для читаемости
plt.yticks(fontsize=9, rotation=0)               # Подписи по Y горизонтально

# Убираем лишние отступы
plt.tight_layout()

# Показываем окно
plt.show()


pairs = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        pairs.append((names[i], names[j], similarity_matrix[i][j]))
pairs.sort(key=lambda x: x[2], reverse=True)

for rank, (name1, name2, score) in enumerate(pairs[:10], 1):
    print(f"{rank}. {name1} ↔ {name2} : {score:.3f}")

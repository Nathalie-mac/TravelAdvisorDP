"""
Контрастное дообучение модели clip-ViT-B-32-multilingual-v1
для рекомендательной системы туристических достопримечательностей.

Цель: достопримечательности одной категории/подтипа — близко в embedding-пространстве,
разных категорий — далеко (по косинусному расстоянию).
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm
from pathlib import Path
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────

class Config:
    # Пути


    # Директория, где лежит train.py
    BASE_DIR = Path(__file__).resolve().parent.parent  # поднимаемся в корень проекта

    DATASET_FILE = BASE_DIR / "data" / "train" / "dataset.json"
    IMAGE_DIR = BASE_DIR / "data" / "train" / "images"
    OUTPUT_DIR   = str(BASE_DIR/ "data" / "models" / "trained_model")

    # Базовая модель
    # Мультиязычная CLIP-модель от sentence-transformers
    MODEL_NAME   = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    # Обучение
    BATCH_SIZE         = 64
    NUM_EPOCHS         = 10
    LEARNING_RATE      = 1e-5
    WEIGHT_DECAY       = 1e-4
    WARMUP_STEPS       = 100
    TEMPERATURE        = 0.07       # τ в InfoNCE / SupCon
    MAX_TEXT_LEN       = 128

    # Стратегия семплирования пар
    # "supcon"  — Supervised Contrastive Loss (рекомендуется)
    # "triplet" — Triplet Margin Loss
    LOSS_TYPE          = "supcon"

    # Веса потерь
    # Итоговый loss = w_cross * L_cross + w_intra * L_intra
    # L_cross — разные категории должны быть далеко
    # L_intra — одна категория+подтип должны быть близко
    W_CROSS            = 1.0
    W_INTRA            = 0.5

    # Hardware
    DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS        = 8
    SEED               = 42


# ──────────────────────────────────────────────
# Датасет
# ──────────────────────────────────────────────

class AttractionDataset(Dataset):
    """
    Ожидаемый формат dataset.json (массив объектов):

    [
      {
        "id": "unique_id",
        "name": "Эрмитаж",
        "description": "Крупнейший художественный музей России...",
        "category": "Культурно-познавательный",
        "subtype": "Музей",
        "images": ["hermitage_1.jpg", "hermitage_2.jpg"]   // имена файлов из IMAGE_DIR
      },
      ...
    ]

    Поля:
      id          — уникальный идентификатор достопримечательности
      name        — название
      description — текстовое описание (желательно 2–5 предложений)
      category    — одна из 9 главных категорий (строго из списка CATEGORIES)
      subtype     — подтип внутри категории (свободная строка, например "Музей")
      images      — список имён файлов изображений (минимум 1)
    """

    CATEGORIES = [
        "Культурно-познавательный",
        "Событийный",
        "Природный",
        "Хайкинг",
        "Спортивный",
        "Экстремальный",
        "Гастрономический",
        "Пляжный",
        "Оздоровительный",
    ]

    def __init__(self, data_file: str, image_dir: str, processor: CLIPProcessor,
                 max_text_len: int = 128):
        self.image_dir   = Path(image_dir)
        self.processor   = processor
        self.max_text_len = max_text_len

        with open(data_file, encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        self.cat2idx  = {c: i for i, c in enumerate(self.CATEGORIES)}

        for item in raw:
            cat = item["category"]
            if cat not in self.cat2idx:
                raise ValueError(f"Неизвестная категория: {cat!r}")
            for img_name in item["images"]:
                self.samples.append({
                    "text"       : item["name"] + ". " + item["description"],
                    "image_path" : str(self.image_dir / img_name),
                    "category"   : cat,
                    "cat_idx"    : self.cat2idx[cat],
                    "subtype"    : item.get("subtype", ""),
                    "attr_id"    : item["id"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Текст
        text_enc = self.processor.tokenizer(
            s["text"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Изображение
        try:
            img = Image.open(s["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        img_enc = self.processor(images=img, return_tensors="pt")

        return {
            "input_ids"      : text_enc["input_ids"].squeeze(0),
            "attention_mask" : text_enc["attention_mask"].squeeze(0),
            "pixel_values"   : img_enc["pixel_values"].squeeze(0),
            "cat_idx"        : torch.tensor(s["cat_idx"], dtype=torch.long),
            "subtype"        : s["subtype"],
            "attr_id"        : s["attr_id"],
        }


# ──────────────────────────────────────────────
# Модель-обёртка
# ──────────────────────────────────────────────

class MultilingualCLIP(nn.Module):
    """
    clip-ViT-B-32-multilingual-v1 состоит из двух частей:
      - CLIP image encoder (из openai/clip-vit-base-patch32)
      - Multilingual text encoder (XLM-RoBERTa + projection head)

    sentence-transformers хранит их раздельно. Здесь мы загружаем
    каждую часть явно, чтобы иметь доступ к параметрам для дообучения.
    """

    def __init__(self, model_name: str):
        super().__init__()

        # Загружаем через HuggingFace CLIP для image-энкодера
        # (веса vision-части совпадают с openai/clip-vit-base-patch32)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Мультиязычный текстовый энкодер через sentence-transformers
        self.text_encoder = SentenceTransformer(model_name)

        # Проецируем оба энкодера в общее 512-мерное пространство
        # (clip-vit-base-patch32 уже даёт 512, text_encoder тоже 512)
        self.embed_dim = 512

    def encode_text(self, input_ids, attention_mask):
        """Получаем текстовые эмбеддинги через мультиязычный энкодер."""
        # sentence-transformers принимает список строк, но мы передаём токены
        # через forward модуля SentenceTransformer напрямую
        features = self.text_encoder({
            "input_ids"     : input_ids,
            "attention_mask": attention_mask,
        })
        embeddings = features["sentence_embedding"]           # (B, 512)
        return F.normalize(embeddings, dim=-1)

    def encode_image(self, pixel_values):
        """Получаем визуальные эмбеддинги через CLIP image encoder."""
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output                  # (B, 768)
        projected = self.clip.visual_projection(pooled)        # (B, 512)
        return F.normalize(projected, dim=-1)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_emb  = self.encode_text(input_ids, attention_mask)
        image_emb = self.encode_image(pixel_values)
        # Итоговый эмбеддинг достопримечательности = среднее текст+изображение
        fused = F.normalize((text_emb + image_emb) / 2.0, dim=-1)
        return fused, text_emb, image_emb


# ──────────────────────────────────────────────
# Функции потерь
# ──────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    SupCon Loss (Khosla et al., 2020).
    Позитивы — объекты из одной категории (опционально: ещё и одного подтипа).
    Негативы — объекты из других категорий.

    loss = -1/|P(i)| * Σ_{p∈P(i)} log [ exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) ]
    """

    def __init__(self, temperature: float = 0.07, use_subtype: bool = False):
        super().__init__()
        self.temperature  = temperature
        self.use_subtype  = use_subtype

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                subtypes: Optional[list] = None) -> torch.Tensor:
        """
        embeddings : (N, D) — L2-нормализованные векторы
        labels     : (N,)   — целочисленные метки категорий
        subtypes   : list[str] длины N — опциональные строки подтипов
        """
        device = embeddings.device
        N = embeddings.size(0)

        # Матрица сходств (N, N)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Маска позитивных пар
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N, N)

        if self.use_subtype and subtypes is not None:
            sub_arr  = np.array(subtypes)
            sub_mask = torch.tensor(
                sub_arr[:, None] == sub_arr[None, :], device=device
            )
            pos_mask = label_mask & sub_mask
            # Если у сэмпла нет ни одного позитива по подтипу — откатываемся к категории
            has_subtype_pos = pos_mask.sum(dim=1) > 1
            pos_mask[~has_subtype_pos] = label_mask[~has_subtype_pos]
        else:
            pos_mask = label_mask

        # Убираем диагональ (i == i)
        diag = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag

        # Если для какого-то сэмпла нет позитивов — пропускаем его
        valid = pos_mask.sum(dim=1) > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # log-softmax по всем "другим" сэмплам (a ≠ i)
        sim_masked = sim.clone()
        sim_masked[diag] = float("-inf")                          # исключаем i==i

        log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)

        # Для каждого сэмпла усредняем log-prob по позитивам
        mean_log_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

        loss = -mean_log_pos[valid].mean()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss.
    Anchor — произвольный сэмпл.
    Positive — случайный сэмпл из той же категории+подтипа.
    Negative — случайный сэмпл из другой категории.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a, b: 1 - F.cosine_similarity(a, b),
            margin=margin,
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                subtypes: Optional[list] = None) -> torch.Tensor:
        device = embeddings.device
        N = embeddings.size(0)
        labels_np = labels.cpu().numpy()

        anchors, positives, negatives = [], [], []

        for i in range(N):
            # Позитив: та же категория
            pos_idx = [j for j in range(N) if j != i and labels_np[j] == labels_np[i]]
            # Негатив: другая категория
            neg_idx = [j for j in range(N) if labels_np[j] != labels_np[i]]

            if not pos_idx or not neg_idx:
                continue

            anchors.append(embeddings[i])
            positives.append(embeddings[random.choice(pos_idx)])
            negatives.append(embeddings[random.choice(neg_idx)])

        if not anchors:
            return torch.tensor(0.0, device=device, requires_grad=True)

        a = torch.stack(anchors)
        p = torch.stack(positives)
        n = torch.stack(negatives)
        return self.loss_fn(a, p, n)


# ──────────────────────────────────────────────
# Тренер
# ──────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Процессор (токенизатор + препроцессинг изображений)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.scaler = GradScaler()
        # Датасет
        dataset = AttractionDataset(
            cfg.DATASET_FILE, cfg.IMAGE_DIR, self.processor, cfg.MAX_TEXT_LEN
        )
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=(cfg.DEVICE == "cuda"),
            drop_last=True,
        )
        print(f"Датасет: {len(dataset)} сэмплов, {len(self.loader)} батчей/эпоха")

        # Модель
        self.model = MultilingualCLIP(cfg.MODEL_NAME).to(cfg.DEVICE)

        # Потери
        if cfg.LOSS_TYPE == "supcon":
            # Основная потеря — сближает одну категорию, разводит разные
            self.loss_cross_cat = SupervisedContrastiveLoss(
                temperature=cfg.TEMPERATURE, use_subtype=False
            )
            # Вспомогательная — дополнительно сближает по подтипу
            self.loss_intra_sub = SupervisedContrastiveLoss(
                temperature=cfg.TEMPERATURE, use_subtype=True
            )
        else:
            self.loss_cross_cat = TripletLoss(margin=0.3)
            self.loss_intra_sub = TripletLoss(margin=0.1)

        # Оптимизатор — дообучаем только проекционные слои и верхние блоки
        params = list(self.model.clip.visual_projection.parameters())
        # Последние 2 transformer-блока vision encoder
        for layer in self.model.clip.vision_model.encoder.layers[-2:]:
            params += list(layer.parameters())
        # Projection head text encoder (последний модуль sentence-transformers)
        for module in list(self.model.text_encoder.modules())[-5:]:
            params += list(module.parameters())

        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
        )

        total_steps = len(self.loader) * cfg.NUM_EPOCHS
        warmup_pct = min(cfg.WARMUP_STEPS / total_steps, 0.3)  # не более 30% на warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=warmup_pct,
        )

    # ── Одна эпоха ──────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.loader, desc=f"Epoch {epoch}"):
            input_ids      = batch["input_ids"].to(self.cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(self.cfg.DEVICE)
            pixel_values   = batch["pixel_values"].to(self.cfg.DEVICE)
            cat_idx        = batch["cat_idx"].to(self.cfg.DEVICE)
            subtypes       = batch["subtype"]          # list[str]

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                fused, text_emb, image_emb = self.model(
                    input_ids, attention_mask, pixel_values
                )
                l_cross = self.loss_cross_cat(fused, cat_idx)
                l_intra = self.loss_intra_sub(fused, cat_idx, subtypes)
                l_align = self._clip_alignment_loss(text_emb, image_emb)
                loss = (
                        self.cfg.W_CROSS * l_cross
                        + self.cfg.W_INTRA * l_intra
                        + 0.1 * l_align
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.loader)

    def _clip_alignment_loss(self, text_emb: torch.Tensor,
                              image_emb: torch.Tensor) -> torch.Tensor:
        """
        Стандартный CLIP-loss: текст i должен быть близок к изображению i.
        Помогает не потерять multimodal alignment при дообучении.
        """
        N = text_emb.size(0)
        logits = torch.matmul(text_emb, image_emb.T) / self.cfg.TEMPERATURE
        targets = torch.arange(N, device=text_emb.device)
        loss_t = F.cross_entropy(logits, targets)
        loss_i = F.cross_entropy(logits.T, targets)
        return (loss_t + loss_i) / 2.0

    # ── Валидация: средняя intra/inter косинусная похожесть ──
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_emb, all_cat = [], []

        for batch in self.loader:
            input_ids      = batch["input_ids"].to(self.cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(self.cfg.DEVICE)
            pixel_values   = batch["pixel_values"].to(self.cfg.DEVICE)
            cat_idx        = batch["cat_idx"]

            fused, _, _ = self.model(input_ids, attention_mask, pixel_values)
            all_emb.append(fused.cpu())
            all_cat.append(cat_idx)

        all_emb = torch.cat(all_emb)
        all_cat = torch.cat(all_cat)
        sim_matrix = torch.matmul(all_emb, all_emb.T)

        N = all_emb.size(0)
        intra_sims, inter_sims = [], []

        for i in range(N):
            for j in range(i + 1, N):
                s = sim_matrix[i, j].item()
                if all_cat[i] == all_cat[j]:
                    intra_sims.append(s)
                else:
                    inter_sims.append(s)

        return {
            "intra_cos_mean": float(np.mean(intra_sims)) if intra_sims else 0.0,
            "inter_cos_mean": float(np.mean(inter_sims)) if inter_sims else 0.0,
            "gap"           : float(np.mean(intra_sims) - np.mean(inter_sims))
                              if intra_sims and inter_sims else 0.0,
        }

    # ── Главный цикл ────────────────────────────
    def train(self):
        best_gap = -float("inf")

        for epoch in range(1, self.cfg.NUM_EPOCHS + 1):
            avg_loss = self.train_epoch(epoch)

            # Валидируем раз в 2 эпохи (дорого считать матрицу попарно)
            if epoch % 2 == 0 or epoch == self.cfg.NUM_EPOCHS:
                metrics = self.evaluate()
                print(
                    f"[Epoch {epoch}] loss={avg_loss:.4f} | "
                    f"intra={metrics['intra_cos_mean']:.4f} | "
                    f"inter={metrics['inter_cos_mean']:.4f} | "
                    f"gap={metrics['gap']:.4f}"
                )

                if metrics["gap"] > best_gap:
                    best_gap = metrics["gap"]
                    self.save(os.path.join(self.cfg.OUTPUT_DIR, "best_model"))
                    print(f"  ✓ Сохранена лучшая модель (gap={best_gap:.4f})")
            else:
                print(f"[Epoch {epoch}] loss={avg_loss:.4f}")

        self.save(os.path.join(self.cfg.OUTPUT_DIR, "final_model"))
        print("Обучение завершено.")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Сохраняем CLIP image encoder
        self.model.clip.save_pretrained(os.path.join(path, "clip"))
        self.processor.save_pretrained(os.path.join(path, "clip"))
        # Сохраняем текстовый энкодер
        self.model.text_encoder.save(os.path.join(path, "text_encoder"))

    def load(self, path: str):
        self.model.clip = CLIPModel.from_pretrained(
            os.path.join(path, "clip")
        ).to(self.cfg.DEVICE)
        self.model.text_encoder = SentenceTransformer(
            os.path.join(path, "text_encoder")
        )


# ──────────────────────────────────────────────
# Инференс: получить эмбеддинг достопримечательности
# ──────────────────────────────────────────────

class AttractionEmbedder:
    """
    Использование дообученной модели для получения эмбеддингов.
    Вектора можно сравнивать через косинусное сходство напрямую.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device    = device
        self.processor = CLIPProcessor.from_pretrained(
            os.path.join(checkpoint_path, "clip")
        )
        self.model     = MultilingualCLIP.__new__(MultilingualCLIP)
        nn.Module.__init__(self.model)
        self.model.clip = CLIPModel.from_pretrained(
            os.path.join(checkpoint_path, "clip")
        ).to(device)
        self.model.text_encoder = SentenceTransformer(
            os.path.join(checkpoint_path, "text_encoder")
        )
        self.model.embed_dim = 512
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str, image_path: str) -> np.ndarray:
        """Возвращает L2-нормализованный вектор размера 512."""
        enc = self.processor.tokenizer(
            text, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        img = Image.open(image_path).convert("RGB")
        img_enc = self.processor(images=img, return_tensors="pt")

        fused, _, _ = self.model(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            img_enc["pixel_values"].to(self.device),
        )
        return fused.squeeze(0).cpu().numpy()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ──────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()
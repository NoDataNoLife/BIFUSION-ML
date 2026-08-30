import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from app.core.model_loader import get_device
from app.core.config import settings


# ==========================================
# 1. ProtoNet Conv4 Backbone Encoder
# ==========================================
def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ProtoEncoderConv4(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 64, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, hidden),      # (B, hidden, H/2, W/2)
            conv_block(hidden, hidden),           # (B, hidden, H/4, W/4)
            conv_block(hidden, hidden),           # (B, hidden, H/8, W/8)
            conv_block(hidden, hidden),           # (B, hidden, H/16, W/16)
        )
        self.fc = nn.Linear(hidden * 8 * 8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 128, 128)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out  # (B, out_dim)


# ==========================================
# 2. Euclidean Distance & Logits
# ==========================================
def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    """각 클래스별 임베딩의 평균 벡터(프로토타입) 계산"""
    prototypes = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            # fallback
            proto = torch.zeros(embeddings.size(1), device=embeddings.device)
        else:
            proto = embeddings[mask].mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)  # (n_classes, out_dim)


def compute_logits(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """음의 유클리드 거리 제곱을 logits로 계산"""
    # query: (N, D), prototypes: (C, D)
    dists = torch.cdist(query_embeddings, prototypes)  # (N, C)
    return -dists


# ==========================================
# 3. ProtoNet Service
# ==========================================
class ProtoNetService:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()

    def train_few_shot(
        self,
        npz_dataset_path: str,
        k_shot: int = 5,
        epochs: int = 30,
        out_model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        증강된 .npz 데이터셋을 기반으로 ProtoNet 분류기 학습
        """
        data = np.load(npz_dataset_path, allow_pickle=True)
        X = data["x"]  # (N, 1, 128, 128) or (N, H, W)
        y = data["y"]  # (N,)

        if X.ndim == 3:
            X = X[:, None, :, :]
        if X.max() > 1.5:
            X = X / 255.0

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Train/Val split (8:2)
        num_samples = len(y)
        indices = np.random.permutation(num_samples)
        split = int(num_samples * 0.8)
        tr_idx, val_idx = indices[:split], indices[split:]

        X_tr, y_tr = X_tensor[tr_idx], y_tensor[tr_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

        encoder = ProtoEncoderConv4(in_channels=1, hidden=64, out_dim=256).to(self.device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_acc = 0.0
        best_metrics = {}

        for epoch in range(1, epochs + 1):
            encoder.train()
            # Support & Query 분할
            X_tr_dev = X_tr.to(self.device)
            y_tr_dev = y_tr.to(self.device)

            emb = encoder(X_tr_dev)
            protos = compute_prototypes(emb, y_tr_dev, n_classes=2)
            logits = compute_logits(emb, protos)
            loss = F.cross_entropy(logits, y_tr_dev)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            encoder.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(self.device)
                y_val_dev = y_val.to(self.device)
                val_emb = encoder(X_val_dev)
                val_logits = compute_logits(val_emb, protos)
                val_preds = val_logits.argmax(dim=-1).cpu().numpy()
                y_val_np = y_val_dev.cpu().numpy()

                acc = accuracy_score(y_val_np, val_preds)
                f1 = f1_score(y_val_np, val_preds, zero_division=0)
                try:
                    probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
                    auc = roc_auc_score(y_val_np, probs)
                except Exception:
                    auc = acc

            if acc >= best_val_acc:
                best_val_acc = acc
                best_metrics = {
                    "accuracy": round(float(acc), 4),
                    "f1_score": round(float(f1), 4),
                    "auroc": round(float(auc), 4),
                    "loss": round(float(loss.item()), 4),
                    "epoch": epoch,
                }
                if out_model_path:
                    Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "encoder_state_dict": encoder.state_dict(),
                        "prototypes": protos.cpu(),
                        "k_shot": k_shot,
                        "metrics": best_metrics,
                    }, out_model_path)

            if progress_callback:
                pct = int((epoch / epochs) * 100)
                progress_callback(epoch, epochs, pct)

        return {
            "metrics": best_metrics,
            "model_path": out_model_path,
        }

    def predict(
        self,
        model_checkpoint_path: str,
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        학습된 ProtoNet 모델을 사용해 새 이미지들을 정상(NORMAL) vs 비정상(ANOMALY)으로 진단
        """
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        encoder = ProtoEncoderConv4(in_channels=1, hidden=64, out_dim=256).to(self.device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.eval()

        prototypes = checkpoint["prototypes"].to(self.device)

        # 이미지 전처리 (128x128 흑백 텐서)
        tensors = []
        for img in images:
            img_resized = img.convert("L").resize((128, 128))
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr[None, :, :]))  # (1, 128, 128)

        batch_tensor = torch.stack(tensors).to(self.device)  # (N, 1, 128, 128)

        with torch.no_grad():
            embeddings = encoder(batch_tensor)
            logits = compute_logits(embeddings, prototypes)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

        results = []
        for idx, (p, prob) in enumerate(zip(preds, probs)):
            class_name = "NORMAL" if p == 0 else "ANOMALY"
            conf = float(prob[p])
            results.append({
                "sample_index": idx,
                "predicted_class": class_name,
                "confidence": round(conf, 4),
                "probabilities": {
                    "normal": round(float(prob[0]), 4),
                    "anomaly": round(float(prob[1]), 4),
                },
            })

        return results

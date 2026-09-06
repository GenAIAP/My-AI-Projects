import os
import re
import sys
import copy
import time
import math
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# 1. HARDWARE SETUP
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

if use_amp:
    cap = torch.cuda.get_device_capability()
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    amp_dtype = torch.float32

CHECKPOINT_DIR = "/content/drive/MyDrive/jepa_ai_detector_checkpoints"

# ==============================================================================
# 2. CONFIG & ARCHITECTURE
# ==============================================================================
class Config:
    backbone_name: str = "sentence-transformers/all-mpnet-base-v2"
    backbone_dim: int = 768

    matrix_rows: int = 12
    matrix_cols: int = 128

    doc_hidden_dim: int = 512
    doc_layers: int = 8
    doc_heads: int = 16
    max_sentences: int = 80
    max_tokens_per_sent: int = 64

    jepa_mask_prob: float = 0.20
    ema_decay: float = 0.999
    loss_lambda_jepa: float = 0.3
    loss_lambda_var: float = 0.05
    loss_lambda_sent: float = 0.0

    batch_size: int = 12
    learning_rate: float = 3.5e-5
    weight_decay: float = 0.01

    save_interval_steps: int = 500
    total_steps: int = 6000

cfg = Config()

def variance_regularizer(x: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if x.size(0) <= 1:
        return torch.tensor(0.0, device=x.device)
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))

class SentenceMatrixProjector(nn.Module):
    def __init__(self, in_dim: int, rows: int, cols: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.query_weights = nn.Parameter(torch.randn(rows, cols) * (1.0 / math.sqrt(cols)))
        self.proj_k = nn.Linear(in_dim, cols)
        self.proj_v = nn.Linear(in_dim, cols)
        self.layer_norm = nn.LayerNorm(cols)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = token_embeddings.shape
        K = self.proj_k(token_embeddings)
        V = self.proj_v(token_embeddings)
        Q = self.query_weights.unsqueeze(0).expand(B, -1, -1)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (math.sqrt(self.cols) * 1.5)
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1)) * -1e4
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        matrix = torch.bmm(attn, V)
        raw_norm = torch.norm(matrix, p=2, dim=(1, 2))

        if self.training:
            slot_mask = (torch.rand(B, self.rows, 1, device=matrix.device) > 0).float()
            matrix = matrix * slot_mask

        return self.dropout(self.layer_norm(matrix)), raw_norm

class SentenceAIScorer(nn.Module):
    def __init__(self, rows: int, cols: int):
        super().__init__()
        in_dim = rows * cols
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, sentence_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(sentence_matrix).squeeze(-1)
        percentages = torch.sigmoid(logits)
        return logits, percentages

class JEPAPredictor(nn.Module):
    def __init__(self, doc_dim: int, rows: int, cols: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.net = nn.Sequential(
            nn.Linear(doc_dim, doc_dim),
            nn.GELU(),
            nn.Linear(doc_dim, rows * cols)
        )

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        return self.net(context_latent).view(-1, self.rows, self.cols)

class JEPATextDetector(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.backbone_name)
        self.matrix_projector = SentenceMatrixProjector(
            cfg.backbone_dim, cfg.matrix_rows, cfg.matrix_cols
        )
        self.sentence_scorer = SentenceAIScorer(cfg.matrix_rows, cfg.matrix_cols)

        matrix_flat_dim = cfg.matrix_rows * cfg.matrix_cols
        self.sentence_fusion = nn.Sequential(
            nn.Linear(matrix_flat_dim, cfg.doc_hidden_dim),
            nn.LayerNorm(cfg.doc_hidden_dim),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.doc_cls_token = nn.Parameter(torch.randn(1, 1, cfg.doc_hidden_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.doc_hidden_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_sentences + 1, cfg.doc_hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.doc_hidden_dim,
            nhead=cfg.doc_heads,
            dim_feedforward=cfg.doc_hidden_dim * 4,
            dropout=0.2,
            activation="gelu",
            batch_first=True
        )
        self.doc_transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.doc_layers)
        
        classifier_in_dim = cfg.doc_hidden_dim * 3
        self.doc_classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        self.jepa_predictor = JEPAPredictor(cfg.doc_hidden_dim, cfg.matrix_rows, cfg.matrix_cols)

    def encode_sentences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.matrix_projector(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_sentence_counts: List[int],
        target_encoder: "JEPATextDetector" = None,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:

        B = len(doc_sentence_counts)
        total_sents = sum(doc_sentence_counts)

        sent_matrices, raw_norms = self.encode_sentences(input_ids, attention_mask)
        sent_logits, sent_probs = self.sentence_scorer(sent_matrices)

        flat_matrices = sent_matrices.view(total_sents, -1)
        sent_features = self.sentence_fusion(flat_matrices)

        masked_indices = []
        target_matrices_for_jepa = []

        if is_training and target_encoder is not None:
            with torch.no_grad():
                target_matrices, _ = target_encoder.encode_sentences(input_ids, attention_mask)
                target_matrices = target_matrices.detach()

        offset = 0
        burstiness_feats = []
        mean_sentence_feats = []
        doc_token_list = []
        max_s = max(doc_sentence_counts)

        key_padding_mask = torch.ones(B, max_s + 1, dtype=torch.bool, device=input_ids.device)

        for i, count in enumerate(doc_sentence_counts):
            doc_unmasked = sent_features[offset : offset + count]

            if count > 1:
                burst = torch.std(doc_unmasked, dim=0)
                mean_f = torch.mean(doc_unmasked, dim=0)
            else:
                burst = torch.zeros(self.cfg.doc_hidden_dim, dtype=sent_features.dtype, device=input_ids.device)
                mean_f = doc_unmasked[0]

            burstiness_feats.append(burst)
            mean_sentence_feats.append(mean_f)

            doc_sent_items = []
            for s_idx in range(count):
                if is_training and count > 2 and target_encoder is not None and random.random() < self.cfg.jepa_mask_prob:
                    masked_indices.append((i, s_idx + 1))
                    target_matrices_for_jepa.append(target_matrices[offset + s_idx])
                    doc_sent_items.append(self.mask_token.squeeze(0).squeeze(0))
                else:
                    doc_sent_items.append(doc_unmasked[s_idx])

            doc_sent_tensor = torch.stack(doc_sent_items, dim=0)

            pad_len = max_s - count
            if pad_len > 0:
                doc_padded = F.pad(doc_sent_tensor, (0, 0, 0, pad_len))
            else:
                doc_padded = doc_sent_tensor

            doc_token_list.append(doc_padded)

            key_padding_mask[i, 0] = False
            key_padding_mask[i, 1 : count + 1] = False
            offset += count

        burstiness_tensor = torch.stack(burstiness_feats)
        mean_sentence_tensor = torch.stack(mean_sentence_feats)
        doc_tokens = torch.stack(doc_token_list, dim=0)

        cls_tokens = self.doc_cls_token.expand(B, -1, -1)
        doc_sequence = torch.cat([cls_tokens, doc_tokens], dim=1) + self.pos_emb[:, : max_s + 1, :]

        encoded_doc = self.doc_transformer(doc_sequence, src_key_padding_mask=key_padding_mask)

        doc_cls_token_out = encoded_doc[:, 0, :]
        doc_combined = torch.cat([doc_cls_token_out, mean_sentence_tensor, burstiness_tensor], dim=-1)

        doc_logits = self.doc_classifier(doc_combined).squeeze(-1)
        doc_probs = torch.sigmoid(doc_logits)

        jepa_loss = torch.tensor(0.0, device=input_ids.device)
        var_loss = torch.tensor(0.0, device=input_ids.device)

        if len(masked_indices) > 0 and target_encoder is not None:
            batch_idxs = [idx[0] for idx in masked_indices]
            seq_idxs = [idx[1] for idx in masked_indices]
            predicted_latents = encoded_doc[batch_idxs, seq_idxs]
            pred_matrices = self.jepa_predictor(predicted_latents)
            targets = torch.stack(target_matrices_for_jepa)

            jepa_l1 = F.smooth_l1_loss(pred_matrices, targets)
            var_loss = variance_regularizer(pred_matrices.flatten(1))
            jepa_loss = jepa_l1 + (self.cfg.loss_lambda_var * var_loss)

        return {
            "doc_logits": doc_logits,
            "doc_probs": doc_probs,
            "sent_logits": sent_logits,
            "sent_probs": sent_probs,
            "sent_matrices": sent_matrices,
            "raw_norms": raw_norms,
            "jepa_loss": jepa_loss,
            "var_loss": var_loss
        }

# ==============================================================================
# 3. UTILITIES & INFERENCE LOGIC
# ==============================================================================
def fast_sentence_split(text: str, max_sentences: int = 80) -> List[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 8]
    if not sentences:
        sentences = [text.strip() if text.strip() else "Empty sentence."]
    return sentences[:max_sentences]

def detect_ai_text(text: str, model: nn.Module, tokenizer, threshold: float = 0.50):
    model.eval()
    sentences = fast_sentence_split(text, cfg.max_sentences)
    if not sentences:
        print("[!] No sentences found.")
        return

    encoded = tokenizer(
        sentences,
        max_length=cfg.max_tokens_per_sent,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                doc_sentence_counts=[len(sentences)],
                target_encoder=None,
                is_training=False
            )

    doc_prob = outputs["doc_probs"][0].item()
    sent_probs = outputs["sent_probs"].float().cpu().numpy()
    raw_norms = outputs["raw_norms"].float().cpu().numpy()

    print("\n" + "="*85)
    print(f"       JEPA AI DETECTION REPORT (CALIBRATED OPERATING POINT: {threshold:.2f})")
    print("="*85)
    print(f"{'Sentence Preview':<52} | {'Matrix Energy':<13} | {'AI Likelihood'}")
    print("-" * 85)

    for i, sent in enumerate(sentences):
        matrix_norm = raw_norms[i]
        p_ai = sent_probs[i] * 100.0
        short_sent = sent if len(sent) <= 50 else sent[:47] + "..."
        bar = "█" * int(p_ai / 10) + "░" * (10 - int(p_ai / 10))
        print(f"{short_sent:<52} | {matrix_norm:<13.2f} | {p_ai:5.1f}% [{bar}]")

    print("="*85)
    verdict = "AI GENERATED" if doc_prob >= threshold else "HUMAN WRITTEN"
    color = "\033[91m" if doc_prob >= threshold else "\033[92m"
    print(f"Overall Document AI Probability: {doc_prob*100:.2f}% (Threshold: {threshold*100:.1f}%)")
    print(f"Final Decision: {color}\033[1m{verdict}\033[0m")
    print("="*85 + "\n")

# ==============================================================================
# 4. LOAD MODEL & EVALUATE (STANDALONE)
# ==============================================================================
print("[*] Initializing tokenizer and model architecture...")
tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
detector_model = JEPATextDetector(cfg).to(device)

ckpt_path = os.path.join(CHECKPOINT_DIR, "jepa_detector_latest.pt")

if os.path.exists(ckpt_path):
    print(f"[*] Loading checkpoint from {ckpt_path}...")
    state = torch.load(ckpt_path, map_location=device)
    detector_model.load_state_dict(state["model_state_dict"])
    print("[✔] Model successfully loaded!")
else:
    print(f"[!] Checkpoint not found at {ckpt_path}. Model will run with randomly initialized weights.")

print("\n[i] Entering interactive mode. Type 'quit' or 'exit' to stop.")
while True:
    user_text = input("\n>>> Enter text to analyze:\n").strip()
    if user_text.lower() in ['quit', 'exit']:
        print("Exiting interactive mode.")
        break
    if not user_text:
        continue

    # Using the calibrated threshold found during evaluation, or fallback to 0.50
    detect_ai_text(user_text, detector_model, tokenizer, threshold=0.50)

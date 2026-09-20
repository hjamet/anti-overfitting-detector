from __future__ import annotations
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Identifiants officiels du Sextuor SOTA (6 Composants Légers SOTA)
DEBERTA_RAID_ID = "desklib/ai-text-detector-v1.01"
MODERNBERT_ID = "GeorgeDrayson/modernbert-ai-detection-raid-mage"
TMR_ROBERTA_ID = "Oxidane/tmr-ai-text-detector"
DEBERTA_ACADEMIC_ID = "desklib/ai-text-detector-academic-v1.01"
XLM_ROBERTA_ID = "yaya36095/xlm-roberta-text-detector"

# Poids nominaux officiels du Sextuor SOTA (Total = 1.00 / 100%)
WEIGHTS = {
    "deberta_raid": 0.23,        # 1. DeBERTa-v3 RAID SOTA (Leader RAID Benchmark)
    "modernbert_long": 0.23,     # 2. ModernBERT Long-Context (8192 ctx, MAGE & RAID)
    "tmr_roberta": 0.18,         # 3. TMR RoBERTa Anti-Paraphrase (Focal Loss & Hard-Negatives)
    "deberta_academic": 0.16,    # 4. DeBERTa-v3 Academic SOTA (Corpus Scientifique)
    "xlm_roberta": 0.11,         # 5. XLM-RoBERTa Multilingue (Cross-lingual Robustness)
    "stylometric_entropy": 0.09, # 6. Moteur Stylométrique & Entropique (Garde-fou non-neural)
}

AI_BUZZWORDS = {
    "furthermore", "moreover", "additionally", "in conclusion", "it is important to note",
    "it is worth noting", "delve", "tapestry", "pivotal", "seamlessly", "multifaceted",
    "paramount", "underscores", "interplay", "holistic", "testament", "crucial",
    "beacon", "foster", "garner", "harness", "intertwined", "linchpin", "myriad",
    "nexus", "nuanced", "plethora", "spearhead", "trailblazing", "unwavering",
    "vibrant", "revolutionize", "game-changer", "meticulously", "realm", "ever-evolving"
}

# Modèle Desklib personnalisé pour régression linéaire sur base DeBERTa
try:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel, AutoConfig, AutoModel

    class DesklibAIDetectionModel(PreTrainedModel):
        """
        Architecture Desklib avec base DeBERTa-v2/v3 et tête de régression linéaire.
        Alignement strict sur le dtype de last_hidden_state pour support natif FP16/BF16.
        """
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            if hasattr(self, "post_init"):
                self.post_init()
            else:
                self.init_weights()

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            logits = self.classifier(pooled_output)
            return type("ModelOutput", (), {"logits": logits})()
except Exception:
    DesklibAIDetectionModel = None


def compute_stylometric_score(text: str) -> Dict[str, Any]:
    """
    Calcule les métriques stylométriques pures :
    Burstiness (CV des longueurs de phrase), TTR, Entropie normalisée et Buzzwords IA.
    """
    raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sents if s.strip()]
    words = re.findall(r'\b[a-zA-ZÀ-ÿ-]+\b', text.lower())
    n_words = len(words)
    n_sents = len(sentences)

    if n_words < 6 or n_sents == 0:
        return {"score": 0.05, "cv_len": 0.80, "ttr": 0.90, "buzzwords_count": 0}

    lens = [len(s.split()) for s in sentences]
    mean_len = sum(lens) / n_sents
    var_len = sum((l - mean_len) ** 2 for l in lens) / n_sents
    std_len = math.sqrt(var_len)
    cv_len = (std_len / mean_len) if mean_len > 0 else 0.0

    unique_words = set(words)
    ttr = len(unique_words) / n_words if n_words > 0 else 0.0

    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    entropy = -sum((c / n_words) * math.log2(c / n_words) for c in counts.values())
    max_ent = math.log2(n_words) if n_words > 1 else 1.0
    norm_entropy = entropy / max_ent if max_ent > 0 else 0.0

    found_buzz = [w for w in words if w in AI_BUZZWORDS]
    buzz_ratio = len(found_buzz) / n_words if n_words > 0 else 0.0

    s_burst = max(0.0, min(1.0, (0.45 - cv_len) / 0.35))
    s_buzz = min(1.0, buzz_ratio * 40.0)
    s_unif = 1.0 if (14.0 <= mean_len <= 26.0 and cv_len < 0.30) else 0.0
    s_ent = max(0.0, min(1.0, 1.0 - abs(norm_entropy - 0.85) * 5.0)) if cv_len < 0.35 else 0.0

    s_stylo = 0.40 * s_burst + 0.35 * s_buzz + 0.15 * s_unif + 0.10 * s_ent
    score = float(round(max(0.0, min(1.0, s_stylo)), 4))
    return {
        "score": score,
        "cv_len": round(cv_len, 3),
        "ttr": round(ttr, 3),
        "buzzwords_count": len(found_buzz)
    }


@dataclass
class EnsembleResult:
    p_ai: float
    p_human: float
    is_ai: bool
    label: str
    components: Dict[str, float]
    weighted_mean: float
    std_dev: float
    disagreement: bool
    veto_triggered: bool

    def to_cli_standard(self) -> str:
        return f"Score P(AI) : {self.p_ai:.4f} ({self.label}) | P(Human) : {self.p_human:.4f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_ai": self.p_ai,
            "p_human": self.p_human,
            "is_ai": self.is_ai,
            "label": self.label,
            "components": self.components,
            "ensemble_stats": {
                "weighted_mean": self.weighted_mean,
                "std_dev": self.std_dev,
                "disagreement": self.disagreement,
                "veto_triggered": self.veto_triggered,
            }
        }


class RAIDEnsembleDetector:
    """
    Gestionnaire et orchestrateur d'inférence pour le Sextuor SOTA léger :
    1. DeBERTa-v3 RAID SOTA (23%)
    2. ModernBERT Long-Context (23%)
    3. TMR RoBERTa Anti-Paraphrase (18%)
    4. DeBERTa-v3 Academic SOTA (16%)
    5. XLM-RoBERTa Multilingue (11%)
    6. Stylométrie & Entropie (9%)
    """
    _instance: Optional[RAIDEnsembleDetector] = None

    def __init__(self, device: Optional[str] = None, hf_token: Optional[str] = None):
        import torch
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        self._deberta_raid_tok = None
        self._deberta_raid_mod = None
        self._modernbert_tok = None
        self._modernbert_mod = None
        self._tmr_tok = None
        self._tmr_mod = None
        self._deberta_acad_tok = None
        self._deberta_acad_mod = None
        self._xlm_tok = None
        self._xlm_mod = None

    @classmethod
    def get_instance(cls, device: Optional[str] = None, hf_token: Optional[str] = None) -> RAIDEnsembleDetector:
        if cls._instance is None:
            cls._instance = RAIDEnsembleDetector(device, hf_token)
        return cls._instance

    def _get_deberta_raid(self):
        if self._deberta_raid_mod is None:
            from transformers import AutoTokenizer
            self._deberta_raid_tok = AutoTokenizer.from_pretrained(DEBERTA_RAID_ID, token=self.hf_token)
            if DesklibAIDetectionModel is not None:
                self._deberta_raid_mod = DesklibAIDetectionModel.from_pretrained(
                    DEBERTA_RAID_ID, token=self.hf_token, torch_dtype=self.torch_dtype
                ).to(self.device).eval()
            else:
                from transformers import AutoModelForSequenceClassification
                self._deberta_raid_mod = AutoModelForSequenceClassification.from_pretrained(
                    DEBERTA_RAID_ID, token=self.hf_token, torch_dtype=self.torch_dtype
                ).to(self.device).eval()
        return self._deberta_raid_tok, self._deberta_raid_mod

    def _get_modernbert(self):
        if self._modernbert_mod is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._modernbert_tok = AutoTokenizer.from_pretrained(MODERNBERT_ID, token=self.hf_token)
            self._modernbert_mod = AutoModelForSequenceClassification.from_pretrained(
                MODERNBERT_ID, token=self.hf_token, torch_dtype=self.torch_dtype
            ).to(self.device).eval()
        return self._modernbert_tok, self._modernbert_mod

    def _get_tmr(self):
        if self._tmr_mod is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tmr_tok = AutoTokenizer.from_pretrained(TMR_ROBERTA_ID, token=self.hf_token)
            self._tmr_mod = AutoModelForSequenceClassification.from_pretrained(
                TMR_ROBERTA_ID, token=self.hf_token, torch_dtype=self.torch_dtype
            ).to(self.device).eval()
        return self._tmr_tok, self._tmr_mod

    def _get_deberta_academic(self):
        if self._deberta_acad_mod is None:
            from transformers import AutoTokenizer
            self._deberta_acad_tok = AutoTokenizer.from_pretrained(DEBERTA_ACADEMIC_ID, token=self.hf_token)
            if DesklibAIDetectionModel is not None:
                self._deberta_acad_mod = DesklibAIDetectionModel.from_pretrained(
                    DEBERTA_ACADEMIC_ID, token=self.hf_token, torch_dtype=self.torch_dtype
                ).to(self.device).eval()
            else:
                from transformers import AutoModelForSequenceClassification
                self._deberta_acad_mod = AutoModelForSequenceClassification.from_pretrained(
                    DEBERTA_ACADEMIC_ID, token=self.hf_token, torch_dtype=self.torch_dtype
                ).to(self.device).eval()
        return self._deberta_acad_tok, self._deberta_acad_mod

    def _get_xlm(self):
        if self._xlm_mod is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._xlm_tok = AutoTokenizer.from_pretrained(XLM_ROBERTA_ID, token=self.hf_token)
            self._xlm_mod = AutoModelForSequenceClassification.from_pretrained(
                XLM_ROBERTA_ID, token=self.hf_token, torch_dtype=self.torch_dtype
            ).to(self.device).eval()
        return self._xlm_tok, self._xlm_mod

    def evaluate(self, text: str, threshold: float = 0.50) -> EnsembleResult:
        import torch

        scores: Dict[str, float] = {}

        # 1. DeBERTa-v3 RAID SOTA (23%)
        try:
            tok_d, mod_d = self._get_deberta_raid()
            inp_d = tok_d(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out_d = mod_d(**inp_d).logits
                p_deb = float(torch.sigmoid(out_d)[0][0].item())
            scores["deberta_raid"] = round(p_deb, 4)
        except Exception as e:
            raise RuntimeError(f"Échec de l'évaluation DeBERTa RAID ({DEBERTA_RAID_ID}) : {e}") from e

        # 2. ModernBERT Long-Context (23%)
        try:
            tok_m, mod_m = self._get_modernbert()
            inp_m = tok_m(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            with torch.no_grad():
                out_m = mod_m(**inp_m).logits
                p_mod = float(torch.softmax(out_m, dim=-1)[0][1].item())
            scores["modernbert_long"] = round(p_mod, 4)
        except Exception as e:
            raise RuntimeError(f"Échec de l'évaluation ModernBERT ({MODERNBERT_ID}) : {e}") from e

        # 3. TMR RoBERTa Anti-Paraphrase (18%)
        try:
            tok_t, mod_t = self._get_tmr()
            inp_t = tok_t(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                out_t = mod_t(**inp_t).logits
                p_tmr = float(torch.softmax(out_t, dim=-1)[0][1].item())
            scores["tmr_roberta"] = round(p_tmr, 4)
        except Exception as e:
            raise RuntimeError(f"Échec de l'évaluation TMR RoBERTa ({TMR_ROBERTA_ID}) : {e}") from e

        # 4. DeBERTa-v3 Academic SOTA (16%)
        try:
            tok_a, mod_a = self._get_deberta_academic()
            inp_a = tok_a(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                out_a = mod_a(**inp_a).logits
                p_acad = float(torch.sigmoid(out_a)[0][0].item())
            scores["deberta_academic"] = round(p_acad, 4)
        except Exception as e:
            raise RuntimeError(f"Échec de l'évaluation DeBERTa Academic ({DEBERTA_ACADEMIC_ID}) : {e}") from e

        # 5. XLM-RoBERTa Multilingue (11%)
        try:
            tok_x, mod_x = self._get_xlm()
            inp_x = tok_x(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                out_x = mod_x(**inp_x).logits
                p_xlm = float(torch.softmax(out_x, dim=-1)[0][1].item())
            scores["xlm_roberta"] = round(p_xlm, 4)
        except Exception as e:
            raise RuntimeError(f"Échec de l'évaluation XLM-RoBERTa ({XLM_ROBERTA_ID}) : {e}") from e

        # 6. Stylométrie & Entropie (9%)
        stylo_res = compute_stylometric_score(text)
        scores["stylometric_entropy"] = stylo_res["score"]

        # Alias de compatibilité ascendante
        scores["deberta_v3_raid"] = scores["deberta_raid"]
        scores["radar_adversarial"] = scores["modernbert_long"]
        scores["electra_combined"] = scores["tmr_roberta"]
        scores["electra_ai"] = scores["tmr_roberta"]
        scores["electra_paraphrased"] = scores["tmr_roberta"]

        # Fusion Mathématique SOTA
        w_mean = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        variance = sum(WEIGHTS[k] * ((scores[k] - w_mean) ** 2) for k in WEIGHTS)
        std_dev = math.sqrt(variance)

        # Veto Conservateur Robuste : si l'un des leaders (DeBERTa RAID ou ModernBERT) est formel (>= 0.90)
        robust_max = max(scores["deberta_raid"], scores["modernbert_long"])
        penalized_mean = w_mean + 0.5 * std_dev

        final_p_ai = min(1.0, max(0.0, max(robust_max if robust_max >= 0.90 else 0.0, penalized_mean)))
        veto_active = (robust_max >= 0.90) and (robust_max > penalized_mean)
        disagreement = std_dev > 0.20

        return EnsembleResult(
            p_ai=round(final_p_ai, 4),
            p_human=round(1.0 - final_p_ai, 4),
            is_ai=final_p_ai >= threshold,
            label="AI" if final_p_ai >= threshold else "Human",
            components=scores,
            weighted_mean=round(w_mean, 4),
            std_dev=round(std_dev, 4),
            disagreement=disagreement,
            veto_triggered=veto_active
        )


# Alias de compatibilité descendante
EnsembleDetectorManager = RAIDEnsembleDetector


def score_text(text: str, threshold: float = 0.50) -> EnsembleResult:
    return RAIDEnsembleDetector.get_instance().evaluate(text, threshold)

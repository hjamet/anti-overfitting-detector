from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional

DEBERTA_MODEL_ID = "desklib/ai-text-detector-v1.01"
RADAR_MODEL_ID = "TrustSafeAI/RADAR-Vicuna-7B"
ELECTRA_MODEL_ID = "vai0511/ai-content-classifier"

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

class EnsembleDetectorManager:
    _instance: Optional[EnsembleDetectorManager] = None

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = self.device == "cuda"
        
        self._deberta_tok = None
        self._deberta_model = None
        self._radar_tok = None
        self._radar_model = None
        self._electra_tok = None
        self._electra_model = None

    @classmethod
    def get_instance(cls) -> EnsembleDetectorManager:
        if cls._instance is None:
            cls._instance = EnsembleDetectorManager()
        return cls._instance

    def _get_deberta(self):
        if self._deberta_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_MODEL_ID)
            self._deberta_model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_ID).to(self.device).eval()
        return self._deberta_tok, self._deberta_model

    def _get_radar(self):
        if self._radar_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._radar_tok = AutoTokenizer.from_pretrained(RADAR_MODEL_ID)
            self._radar_model = AutoModelForSequenceClassification.from_pretrained(RADAR_MODEL_ID).to(self.device).eval()
        return self._radar_tok, self._radar_model

    def _get_electra(self):
        if self._electra_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._electra_tok = AutoTokenizer.from_pretrained(ELECTRA_MODEL_ID)
            self._electra_model = AutoModelForSequenceClassification.from_pretrained(ELECTRA_MODEL_ID).to(self.device).eval()
        return self._electra_tok, self._electra_model

    def evaluate(self, text: str, threshold: float = 0.50) -> EnsembleResult:
        import torch
        import torch.nn.functional as F

        scores = {}

        # 1. DeBERTa-v3 Large (Index 1 = AI)
        try:
            tok_d, mod_d = self._get_deberta()
            inp_d = tok_d(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                logits_d = mod_d(**inp_d).logits
                p_deb = float(F.softmax(logits_d, dim=-1)[:, 1].item())
            scores["deberta_v3_raid"] = round(p_deb, 4)
        except Exception as e:
            scores["deberta_v3_raid"] = 0.50

        # 2. RADAR RoBERTa-Large (Index 0 = AI !)
        try:
            tok_r, mod_r = self._get_radar()
            inp_r = tok_r(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                logits_r = mod_r(**inp_r).logits
                p_rad = float(F.softmax(logits_r, dim=-1)[:, 0].item())
            scores["radar_adversarial"] = round(p_rad, 4)
        except Exception as e:
            scores["radar_adversarial"] = 0.50

        # 3. ELECTRA 3-Class (0: Human, 1: AI, 2: Paraphrased)
        try:
            tok_e, mod_e = self._get_electra()
            inp_e = tok_e(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.inference_mode():
                logits_e = mod_e(**inp_e).logits
                probs_e = F.softmax(logits_e, dim=-1)[0]
                p_ai_e = float(probs_e[1].item())
                p_para_e = float(probs_e[2].item())
            p_electra_combined = p_ai_e + 0.5 * p_para_e
            scores["electra_ai"] = round(p_ai_e, 4)
            scores["electra_paraphrased"] = round(p_para_e, 4)
            scores["electra_combined"] = round(min(1.0, p_electra_combined), 4)
        except Exception as e:
            scores["electra_combined"] = 0.50

        # 4. Mathematical Fusion
        weights = {"deberta_v3_raid": 0.45, "radar_adversarial": 0.35, "electra_combined": 0.20}
        w_mean = sum(scores.get(k, 0.5) * weights[k] for k in weights)
        variance = sum(weights[k] * ((scores.get(k, 0.5) - w_mean) ** 2) for k in weights)
        std_dev = math.sqrt(variance)

        # Conservative Robust-Veto: Max of robust detectors vs penalized mean
        robust_max = max(scores.get("deberta_v3_raid", 0.0), scores.get("radar_adversarial", 0.0))
        penalized_mean = w_mean + 0.5 * std_dev
        
        final_p_ai = min(1.0, max(0.0, max(robust_max, penalized_mean)))
        veto_active = robust_max > penalized_mean
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

def score_text(text: str, threshold: float = 0.50) -> EnsembleResult:
    return EnsembleDetectorManager.get_instance().evaluate(text, threshold)

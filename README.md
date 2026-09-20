# 🛡️ Anti-Overfitting AI Detector Ensemble (SOTA Lightweight Sextet)

Robust, state-of-the-art AI text detection ensemble combining 6 orthogonal lightweight neural and statistical components from the **RAID Benchmark (ACL 2024)** to eliminate adversarial overfitting, synonym-swap bypasses, and human-preamble evasion without heavy causal LLMs.

## 🚀 Key Architectural Pillars (SOTA Lightweight Sextet)

1. **DeBERTa-v3 RAID SOTA (23% Weight)** (`desklib/ai-text-detector-v1.01`):
   - N°1 Leader on the RAID Benchmark.
   - Disentangled attention mechanism ($Q_c K_p^T + Q_p K_c^T$) immune to local word order permutations.
2. **ModernBERT Long-Context (23% Weight)** (`GeorgeDrayson/modernbert-ai-detection-raid-mage`):
   - Native 8192-token context window with rotary positional embeddings (RoPE).
   - Fine-tuned on MAGE and RAID benchmark datasets for long-form document coherence.
3. **TMR RoBERTa Anti-Paraphrase (18% Weight)** (`Oxidane/tmr-ai-text-detector`):
   - Trained with Focal Loss and Hard-Negative Mining against iterative paraphrasing and spin attacks.
4. **DeBERTa-v3 Academic SOTA (16% Weight)** (`desklib/ai-text-detector-academic-v1.01`):
   - Specialized for scientific literature, research papers, and technical prose.
5. **XLM-RoBERTa Multilingual (11% Weight)** (`yaya36095/xlm-roberta-text-detector`):
   - Cross-lingual robustness and multilingual semantic transfer across 100+ languages.
6. **Stylometric & Shannon Entropy Engine (9% Weight)**:
   - Non-neural safeguard computing sentence burstiness (CV), type-token ratio (TTR), Shannon entropy, and AI buzzword frequency.

## 📐 Mathematical Fusion: Conservative Robust-Veto

$$\mathcal{P}_{\text{ensemble}}(x) = \max \left( \max_{\mathcal{R}} P_k(x), \; \sum_{k} \omega_k P_k(x) + 0.5 \cdot \sigma_P(x) \right)$$

Where $\sum_{k=1}^6 \omega_k = 1.00$. If a proven robust anchor detector ($k \in \{\text{DeBERTa RAID}, \text{ModernBERT}\}$) flags the text as AI ($P \ge 0.90$), its veto applies immediately, preventing score dilution.

## ⚡ Hardware Efficiency & VRAM Footprint
- **Total VRAM Footprint** : $< 3.0 \text{ GB}$ in FP16 / BF16 (runs comfortably on 4GB-12GB GPUs).
- **CPU Compatibility** : Sub-second inference on multi-threaded CPUs with zero external GPU dependency.
- **Fail-Fast Doctrine** : Strict Zero-Trust exception handling with no silent fallbacks.

## 💻 Python & CLI Usage

### Python API
```python
from anti_overfitting_detector.ensemble import RAIDEnsembleDetector, score_text

# Quick scoring
result = score_text("Your text to evaluate here...")
print(result.to_cli_standard())
print(result.components)

# Object-oriented API with custom device
detector = RAIDEnsembleDetector.get_instance(device="cuda")
res = detector.evaluate("Scientific paper abstract...")
```

### CLI
```bash
# Standard scoring (100% backward-compatible output)
ai-detector score "Your text to evaluate here..."

# Detailed breakdown with sub-models and uncertainty penalty
ai-detector score "Your text here..." --verbose

# Structured JSON output
ai-detector score "Your text here..." --json
```

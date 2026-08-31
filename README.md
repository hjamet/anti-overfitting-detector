# 🛡️ Anti-Overfitting AI Detector Ensemble

Robust, state-of-the-art AI text detection ensemble combining multiple orthogonal neural architectures from the **RAID Benchmark (ACL 2024)** to eliminate adversarial overfitting, synonym-swap bypasses, and human-preamble evasion.

## 🚀 Key Architectural Pillars

1. **DeBERTa-v3-Large (45% Weight)** (`desklib/ai-text-detector-v1.01`):
   - N°1 Leader on the RAID Benchmark.
   - Disentangled attention mechanism ($Q_c K_p^T + Q_p K_c^T$) immune to local word order permutations.
2. **RADAR RoBERTa-Large (35% Weight)** (`TrustSafeAI/RADAR-Vicuna-7B`):
   - Trained via adversarial minimax game directly against paraphrasers.
   - Gradients closed to rewrite directions.
3. **ELECTRA 3-Class Paraphrase Discriminator (20% Weight)** (`vai0511/ai-content-classifier`):
   - Multi-class categorization: `Human`, `AI-Generated`, `Paraphrased`.
   - Intercepts artificial humanization and spin techniques.

## 📐 Mathematical Fusion: Conservative Robust-Veto

$$\\mathcal{P}_{\\text{ensemble}}(x) = \\max \\left( \\max_{\\mathcal{R}} P_k(x), \\; \\sum_{k} \\omega_k P_k(x) + 0.5 \\cdot \\sigma_P(x) \\right)$$

If a proven robust detector ($k \\in \\{\\text{DeBERTa}, \\text{RADAR}\\}$) flags the text as AI ($P \\ge 0.90$), its veto applies immediately, preventing score dilution.

## ⚡ GPU Acceleration
Automatically detects NVIDIA CUDA GPUs (e.g. RTX 3060 12GB) and executes with FP16 autocast for sub-50ms inference. Fallbacks seamlessly to multi-threaded CPU mode.

## 💻 CLI Usage

```bash
# Standard scoring (100% backward-compatible output)
ai-detector score "Your text to evaluate here..."

# Detailed breakdown with sub-models and uncertainty penalty
ai-detector score "Your text here..." --verbose

# Structured JSON output
ai-detector score "Your text here..." --json
```

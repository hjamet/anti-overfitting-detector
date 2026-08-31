import sys
import argparse
import json
from .ensemble import score_text, EnsembleDetectorManager

def main():
    parser = argparse.ArgumentParser(description="Anti-Overfitting AI Detector Ensemble (RAID Benchmark SOTA)")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # score command
    score_p = subparsers.add_parser("score", help="Score text for AI probability")
    score_p.add_argument("text", nargs="?", default="", help="Text to evaluate")
    score_p.add_argument("--threshold", type=float, default=0.50, help="Classification threshold (default: 0.50)")
    score_p.add_argument("--verbose", "-v", action="store_true", help="Display full detailed breakdown")
    score_p.add_argument("--json", action="store_true", help="Output JSON structure")
    
    args = parser.parse_args()
    
    if args.command == "score":
        text = args.text
        if not text and not sys.stdin.isatty():
            text = sys.stdin.read()
        if not text:
            print("Error: text is required.", file=sys.stderr)
            sys.exit(1)
            
        res = score_text(text, args.threshold)
        
        if args.json:
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        elif args.verbose:
            print("=" * 65)
            print("🛡️ ÉVALUATION DE DÉTECTION IA — ENSEMBLE ANTI-OVERFITTING (RAID SOTA)")
            print("=" * 65)
            print("Composants Individuels :")
            print(f"  • DeBERTa-v3 Large [RAID N°1]       : P(AI) = {res.components.get('deberta_v3_raid', 0.0):.4f}")
            print(f"  • RADAR RoBERTa   [Minimax]        : P(AI) = {res.components.get('radar_adversarial', 0.0):.4f}")
            print(f"  • ELECTRA 3-Class [Paraphrase]     : P(AI) = {res.components.get('electra_ai', 0.0):.4f} | P(Para) = {res.components.get('electra_paraphrased', 0.0):.4f}")
            print("-" * 65)
            print("Métriques de Fusion :")
            print(f"  • Moyenne Pondérée : {res.weighted_mean:.4f} | Écart-Type (σ) : {res.std_dev:.4f}")
            if res.veto_triggered:
                print("  • ⚠️ Veto Robuste Actif (Détection formelle par DeBERTa ou RADAR)")
            print("-" * 65)
            print(res.to_cli_standard())
            print("=" * 65)
        else:
            print(res.to_cli_standard())
            print(f"  - DeBERTa-v3 RAID : {res.components.get('deberta_v3_raid', 0.0):.4f}")
            print(f"  - RADAR Minimax   : {res.components.get('radar_adversarial', 0.0):.4f}")
            print(f"  - ELECTRA 3-Class : {res.components.get('electra_combined', 0.0):.4f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

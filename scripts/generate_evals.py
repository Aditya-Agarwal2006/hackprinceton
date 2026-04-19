#!/usr/bin/env python3
"""Generate evaluation dataset using Gemini API for UDC testing."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import from app/
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.gemini_client import generate_eval_dataset, _MODEL_NAME, _FALLBACK_MODEL


DOMAINS = {
    "medical": 20,
    "scientific": 20,
    "general_knowledge": 20,
    "clinical_trials": 10,
}


def main():
    results = {
        "generated_at": datetime.now().isoformat(),
        "primary_model": _MODEL_NAME,
        "fallback_model": _FALLBACK_MODEL,
        "length_threshold": "20%",
        "domains": {},
        "total_pairs": 0,
        "pairs": [],
    }

    for domain, count in DOMAINS.items():
        print(f"Generating {count} pairs for domain: {domain}...")
        pairs = generate_eval_dataset(domain, count)
        rewrites = sum(1 for p in pairs if "(rewrite)" in p.get("generator_model", ""))
        results["domains"][domain] = {
            "requested": count,
            "generated": len(pairs),
            "rewrites": rewrites,
        }
        results["pairs"].extend(pairs)
        print(f"  Got {len(pairs)} / {count} pairs ({rewrites} rewrites)")

    results["total_pairs"] = len(results["pairs"])

    # Deduplicate by prompt text (keep first occurrence)
    seen_prompts = set()
    deduped = []
    dupes = 0
    for pair in results["pairs"]:
        key = pair["prompt"].strip().lower()
        if key in seen_prompts:
            dupes += 1
            continue
        seen_prompts.add(key)
        deduped.append(pair)
    results["pairs"] = deduped
    results["total_pairs"] = len(deduped)
    results["duplicates_removed"] = dupes

    # Recompute domain stats from the final deduplicated set so the metadata
    # matches what is actually saved.
    for domain, info in results["domains"].items():
        domain_pairs = [p for p in deduped if p.get("domain") == domain]
        info["generated"] = len(domain_pairs)
        info["rewrites"] = sum(1 for p in domain_pairs if "(rewrite)" in p.get("generator_model", ""))

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gemini_eval_pairs.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {results['total_pairs']} total pairs to {out_path}")
    if dupes:
        print(f"  Removed {dupes} duplicate prompts")
    for domain, info in results["domains"].items():
        print(f"  {domain}: {info['generated']}/{info['requested']} ({info['rewrites']} rewrites)")


if __name__ == "__main__":
    main()

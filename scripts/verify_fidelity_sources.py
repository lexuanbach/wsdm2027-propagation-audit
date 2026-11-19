"""Verify the local side of the source-comparison ledger without network or training."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[1]

def main():
    ledger = json.loads((ROOT / "FIDELITY-SOURCES.json").read_text())
    failures = [name for name, expected in ledger["local_files"].items()
                if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected]
    if failures:
        raise SystemExit("Fidelity source mismatch: " + ", ".join(failures))
    print(f"PASS: {len(ledger['local_files'])} local source/output bindings; upstream descriptors only, no training rerun.")

if __name__ == "__main__":
    main()

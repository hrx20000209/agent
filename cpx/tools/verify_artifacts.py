"""Verify archived branch files; does not touch devices or model services."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'ARTIFACTS.json').read_text())
for row in manifest:
 p=root/row['path']
 assert p.stat().st_size==row['bytes'],p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
print(f"OK: {len(manifest)} archived files")

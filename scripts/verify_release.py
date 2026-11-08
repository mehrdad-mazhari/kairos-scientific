"""Read-only hash verification; never invokes a sampler or simulator."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=root/'reproducibility/original'
frozen=json.loads((original/'results/preconfirmatory_checks_v1/freeze_manifest.json').read_text())
for rel,digest in frozen.items():
    assert sha(original/'results/preconfirmatory_checks_v1'/rel)==digest,rel
manifest_path=root/'docs/github_release_manifest.json'
if manifest_path.exists():
    for rel,digest in json.loads(manifest_path.read_text())['files'].items():
        assert sha(root/rel)==digest,rel
print(f'PASS: {len(frozen)} frozen hashes; release manifest verified if present. No devices generated.')

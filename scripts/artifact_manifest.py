"""Build or verify SHA-256 hashes of the small reproducibility artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    target = Path('results/artifact_manifest.json')
    if args.verify:
        saved = json.loads(target.read_text())
        bad = [name for name, expected in saved['sha256'].items()
               if not Path(name).is_file() or digest(Path(name)) != expected]
        if bad:
            raise SystemExit('Missing or changed artifacts: ' + ', '.join(bad))
        print(f'Verified {len(saved["sha256"])} artifacts')
        return
    paths = set()
    for directory in ['configs', 'src', 'scripts', 'tests', 'docs', 'results/tables', 'results/figures']:
        paths.update(p for p in Path(directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix in {'.py', '.md', '.yaml', '.csv', '.json', '.png'})
    for pattern in ['README.md', 'LOG.md', 'requirements*.txt', 'results/validation*.xml',
                    'results/clean_smoke_comparison.json', 'results/workflow_status.json',
                    'results/runs/*/*/manifest.json', 'results/runs/*/*/status.json']:
        paths.update(Path('.').glob(pattern))
    record = dict(scope='Source, configs, documents, tables, figures, validation and run metadata; large caches and raw data excluded',
                  sha256={p.as_posix(): digest(p) for p in sorted(paths)})
    target.write_text(json.dumps(record, indent=2))
    print(f'Hashed {len(paths)} artifacts')


if __name__ == '__main__':
    main()

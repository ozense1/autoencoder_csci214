"""Read BEIR-format data and write small metadata files atomically."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from config import QREL_SPLITS


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_dataset(path, split='test'):
    """Return documents, queries and qrels for one judgment split.

    split='test' (default) reads qrels/test.tsv, the held-out final evaluation set.
    split='dev' reads qrels/train.tsv, used only for relevance-based model selection.
    """
    if split not in QREL_SPLITS:
        raise ValueError(f'Unknown qrels split {split!r}; choose one of {sorted(QREL_SPLITS)}.')
    path = Path(path)
    def records(filename):
        with (path / filename).open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        ids = [str(r['_id']) for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {filename}')
        return rows
    documents = records('corpus.jsonl')
    queries = {str(row['_id']): row['text'] for row in records('queries.jsonl')}
    qrels = {}
    with (path / 'qrels' / QREL_SPLITS[split]).open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            qrels.setdefault(row['query-id'], {})[row['corpus-id']] = float(row['score'])
    corpus_ids = {str(d['_id']) for d in documents}
    if not set(qrels).issubset(queries):
        raise ValueError('Some judged queries are missing from queries.jsonl')
    if any(doc_id not in corpus_ids for rels in qrels.values() for doc_id in rels):
        raise ValueError('Some judged documents are missing from corpus.jsonl')
    return documents, queries, qrels


def portable_path(path, artifact_dir):
    """Store resource locations relative to the registry's artifact directory."""
    return Path(os.path.relpath(Path(path).expanduser().resolve(), Path(artifact_dir).resolve())).as_posix()


def resolve_resource(artifact_dir, stored_path):
    """Resolve portable registry paths independently of the working directory.

    Absolute entries from older registries remain readable; prepare rewrites
    resource locations using the portable format.
    """
    path = Path(stored_path).expanduser()
    return (path if path.is_absolute() else Path(artifact_dir) / path).resolve()

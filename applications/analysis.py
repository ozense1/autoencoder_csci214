"""Report artifacts from completed experiments
    nothing is retrained.

Reads the experiment summaries written by experiment.py and produces 
in artifacts/analysis/<name>/:

    loss_components.png                                                         validation MSE / cosine / geometry / contractive per run
    train_vs_validation.png                                                     weighted training and validation objective
    gamma_sweep.png                                                             dev nDCG@10 and validation geometry error versus gamma
    coordinate_std.png                                                          sorted per-coordinate std of raw z (collapse check)
    summary.md / summary.json
        - collapse diagnostics for every analysed run (raw and normalized z)
        - exact cosine search versus Chroma HNSW (same metrics, top-100 overlap)
        - paired per-seed and per-query (bootstrap) comparison, proposed vs ablation
        - storage and parameter counts
    query_examples.md                                                           dev queries where the proposed method helps / hurts

Usage (names are the --name values given to experiment.py):
    python applications/analysis.py --final final --sweep gamma-sweep-3seed --gamma 1
"""
import argparse
from pathlib import Path
import statistics

import numpy as np
import torch

from config import ARTIFACT_DIR, METHODS
from data_utils import load_dataset, read_json, resolve_resource, write_json
from embeddings import normalize_rows
from matryoshka import truncate
from metrics import METRIC_NAMES, query_metrics
from retrieval import query_embedding
from storage import load_registry
from training import load_encoder

FLOAT_BYTES = 4


# ---------------------------------------------------------------- loading

def model_entries(artifact_dir):
    """Every trained run on disk, keyed by its index fingerprint."""
    entries = {}
    for path in sorted((artifact_dir / 'models').glob('*/entry.json')):
        entry = read_json(path)
        entries[entry['index_fingerprint']] = entry
    return entries


def experiment(artifact_dir, split, name):
    path = artifact_dir / 'experiments' / split / f'{name}.json'
    if not path.exists():
        raise SystemExit(f'Missing {path}; run experiment.py --name {name} --split {split} first.')
    return read_json(path)


def find_report(artifact_dir, split, method, index_fp, hint=None):
    """Locate the evaluation report containing this method/index (per-query rankings)."""
    if hint and Path(hint).exists():
        return read_json(hint)
    folder = artifact_dir / 'evaluations' / ('' if split == 'test' else split)
    for path in sorted(folder.glob('*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        report = read_json(path)
        values = report.get('methods', {}).get(method)
        if values and values['index_fingerprint'] == index_fp and report.get('split', 'test') == split:
            return report
    return None


def run_label(row):
    return row['label']


# ---------------------------------------------------------------- statistics

def effective_dimension(matrix):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(centered, rowvar=False)), 0, None)
    return float(eigenvalues.sum() ** 2 / max((eigenvalues ** 2).sum(), 1e-30))


def collapse_statistics(raw, normalized):
    raw_std = raw.std(axis=0)
    unit_std = normalized.std(axis=0)
    sample = normalized[:2000]
    similarity = sample @ sample.T
    size = len(sample)
    return {
        'raw_mean_norm': float(np.linalg.norm(raw, axis=1).mean()),
        'raw_mean_coord_std': float(raw_std.mean()),
        'raw_min_coord_std': float(raw_std.min()),
        'raw_coords_std_lt_1e-3': int((raw_std < 1e-3).sum()),
        'saturated_fraction_abs_gt_0.99': float((np.abs(raw) > .99).mean()),
        'unit_mean_coord_std': float(unit_std.mean()),
        'unit_min_coord_std': float(unit_std.min()),
        'mean_pairwise_cosine': float((similarity.sum() - np.trace(similarity)) / (size * (size - 1))),
        'effective_dimension': effective_dimension(normalized),
    }, np.sort(raw_std)[::-1]


def bootstrap_mean_ci(values, resamples=10000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


# ---------------------------------------------------------------- representations

class Spaces:
    """Document and query vectors for each analysed run, computed as retrieval does."""

    def __init__(self, artifact_dir, baseline, documents):
        self.artifact_dir = artifact_dir
        self.baseline = baseline
        self.ids = [str(d['_id']) for d in documents]
        self.embeddings = np.load(resolve_resource(artifact_dir, baseline['embedding_file']))
        self.engine = None

    def embedder(self):
        if self.engine is None:
            from embeddings import Embedder
            model_path = resolve_resource(self.artifact_dir, self.baseline['model_path'])
            self.engine = Embedder(model_path, **self.baseline['runtime'])
        return self.engine

    def queries(self, texts):
        return np.stack([query_embedding(text, self.baseline, self.artifact_dir, self.embedder)[0] for text in texts])

    def transform(self, row, entry, vectors):
        """Return (raw, normalized) representations for a [n, 768] input batch."""
        if row['method'] == 'baseline':
            return vectors, normalize_rows(vectors)
        if row['method'].startswith('matryoshka'):
            dimension = row['dimension']
            return vectors[:, :dimension], truncate(vectors, dimension)
        model = load_encoder(self.artifact_dir, entry)
        with torch.no_grad():
            raw = torch.cat([model.encode(chunk) for chunk in torch.from_numpy(np.ascontiguousarray(vectors)).split(512)]).numpy()
        return raw, normalize_rows(raw)

    def close(self):
        if self.engine:
            self.engine.close()


# ---------------------------------------------------------------- plots

def plot_curves(rows, entries, artifact_dir, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trained, seen, aliases = [], {}, {}
    for row in (r for r in rows if r['index_fingerprint'] in entries):
        # Runs with identical histories (e.g. gamma=0 vs mse_cosine) are drawn once.
        history = read_json(artifact_dir / entries[row['index_fingerprint']]['history'])
        key = (row['seed'], tuple(round(h['validation_total'], 12) for h in history))
        if key in seen:
            aliases.setdefault(seen[key], set()).add(run_label(row))
            continue
        seen[key] = run_label(row)
        trained.append(row)
    rename = {label: f"{label} = {' = '.join(sorted(names))}" for label, names in aliases.items()}
    trained = [{**r, 'label': rename.get(run_label(r), run_label(r))} for r in trained]
    labels = list(dict.fromkeys(run_label(r) for r in trained))
    colors = {label: f'C{i}' for i, label in enumerate(labels)}
    styles = ['-', '--', ':', '-.']
    seeds = sorted({r['seed'] for r in trained}, key=str)
    components = ['mse', 'cosine', 'geometry', 'contractive']
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, component in zip(axes.flat, components):
        for row in trained:
            entry = entries[row['index_fingerprint']]
            history = read_json(artifact_dir / entry['history'])
            key = f'validation_{component}'
            if key not in history[0]:
                continue
            epochs = [h['epoch'] for h in history]
            values = [h[key] for h in history]
            style = styles[seeds.index(row['seed']) % len(styles)]
            axis.plot(epochs, values, style, color=colors[run_label(row)], linewidth=1.2,
                        label=f"{run_label(row)} s{row['seed']}")
            best = entry['best_epoch']
            axis.plot(best, history[best - 1][key], 'o', color=colors[run_label(row)], markersize=4)
        axis.set_title(f'Validation {component} (unweighted)')
        axis.set_xlabel('Epoch')
        if component in ('geometry', 'mse'):
            axis.set_yscale('log')
    handles, names = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', ncol=3, fontsize=8)
    figure.suptitle('Loss components on held-out documents (dots = selected checkpoint)')
    figure.tight_layout(rect=(0, .1, 1, .96))
    figure.savefig(out / 'loss_components.png', dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 3.6), squeeze=False)
    for axis, label in zip(axes[0], labels):
        row = next(r for r in trained if run_label(r) == label)
        history = read_json(artifact_dir / entries[row['index_fingerprint']]['history'])
        epochs = [h['epoch'] for h in history]
        axis.plot(epochs, [h['train_total'] for h in history], label='train')
        axis.plot(epochs, [h['validation_total'] for h in history], label='validation')
        axis.axvline(entries[row['index_fingerprint']]['best_epoch'], color='grey', linestyle=':')
        axis.set_title(f"{label} (seed {row['seed']})", fontsize=9)
        axis.set_xlabel('Epoch')
        axis.legend(fontsize=8)
    axes[0][0].set_ylabel('Weighted objective')
    figure.tight_layout()
    figure.savefig(out / 'train_vs_validation.png', dpi=150)
    plt.close(figure)


def plot_sweep(sweep, entries, artifact_dir, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = [r for r in sweep['runs'] if r.get('gamma') is not None]
    if not runs:
        return []
    gammas = sorted({r['gamma'] for r in runs})
    table = []
    for gamma in gammas:
        group = [r for r in runs if r['gamma'] == gamma]
        scores = [r['nDCG@10'] for r in group]
        geometry = []
        for r in group:
            entry = entries.get(r['index_fingerprint'])
            if entry:
                history = read_json(artifact_dir / entry['history'])
                geometry.append(history[entry['best_epoch'] - 1].get('validation_geometry', float('nan')))
        table.append({'gamma': gamma, 'runs': len(group), 'nDCG@10': statistics.fmean(scores),
                        'nDCG@10_std': statistics.stdev(scores) if len(scores) > 1 else 0.,
                        'validation_geometry': statistics.fmean(geometry) if geometry else float('nan')})
    references = {r['label']: r['nDCG@10'] for r in sweep['runs'] if r.get('gamma') is None}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Evenly spaced categorical positions: gamma includes 0, so a log axis is misleading.
    x = list(range(len(table)))
    ticks = [f"{t['gamma']:g}" for t in table]
    axes[0].errorbar(x, [t['nDCG@10'] for t in table], yerr=[t['nDCG@10_std'] for t in table], marker='o', capsize=4, label='MSE + cosine + geometry')
    for i, (label, value) in enumerate(references.items()):
        axes[0].axhline(value, linestyle='--', color=f'C{i + 1}', label=label)
    axes[0].set_xticks(x, ticks)
    axes[0].set_xlabel('gamma')
    axes[0].set_ylabel('dev nDCG@10 (mean ± std over seeds)')
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [t['validation_geometry'] for t in table], marker='o')
    axes[1].set_xticks(x, ticks)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('gamma')
    axes[1].set_ylabel('validation geometry error at selected epoch')
    figure.tight_layout()
    figure.savefig(out / 'gamma_sweep.png', dpi=150)
    plt.close(figure)
    return table


def plot_coordinate_std(spectra, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Merge runs with identical spectra (e.g. gamma=0 and mse_cosine) into one line.
    merged = []
    for label, values in spectra.items():
        for item in merged:
            if item['values'].shape == values.shape and np.allclose(item['values'], values, rtol=1e-6, atol=1e-9):
                item['label'] += f' = {label}'
                break
        else:
            merged.append({'label': label, 'values': values})
    for i, item in enumerate(merged):
        item['color'] = f'C{i}'
    # Zoom panel: the healthy trained encoders, whose curves sit close together.
    medians = {item['label']: float(np.median(item['values'])) for item in merged}
    top = max(medians.values())
    zoomed = [item for item in merged if medians[item['label']] >= .6 * top]

    figure, (full, zoom) = plt.subplots(1, 2, figsize=(13, 4.8))
    for item in merged:
        x = np.arange(1, len(item['values']) + 1)
        full.plot(x, np.maximum(item['values'], 1e-8), color=item['color'], label=item['label'])
    full.set_yscale('log')
    full.set_title('All methods (log scale)')
    full.set_xlabel('Latent coordinate (sorted by std)')
    full.set_ylabel('Std of raw z across documents')
    if zoomed:
        low = min(item['values'].min() for item in zoomed)
        high = max(item['values'].max() for item in zoomed)
        pad = .05 * (high - low)
        full.axhspan(low - pad, high + pad, color='grey', alpha=.12)
        for item in zoomed:
            x = np.arange(1, len(item['values']) + 1)
            zoom.plot(x, item['values'], color=item['color'], linewidth=1.6)
        zoom.set_ylim(low - pad, high + pad)
        zoom.set_title('Zoom on trained encoders (linear scale, shaded band at left)')
        zoom.set_xlabel('Latent coordinate (sorted by std)')
        zoom.set_ylabel('Std of raw z across documents')
        zoom.grid(alpha=.3)
    figure.suptitle('Per-coordinate variation of raw latent codes (collapse check)')
    handles, names = full.get_legend_handles_labels()
    figure.legend(handles, names, loc='center left', bbox_to_anchor=(1.0, .5), fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(out / 'coordinate_std.png', dpi=150, bbox_inches='tight')
    plt.close(figure)


# ---------------------------------------------------------------- query examples

def query_examples(sweep, gamma, artifact_dir, queries, qrels, titles, count=2):
    runs = sweep['runs']
    proposed = {r['seed']: r for r in runs if r.get('gamma') == gamma}
    ablation = {r['seed']: r for r in runs if r.get('gamma') == 0}
    seeds = sorted(set(proposed) & set(ablation))
    if not seeds:
        return None, 'No paired proposed/ablation dev runs found in the sweep.'
    per_seed = []
    for seed in seeds:
        reports = {}
        for name, row in (('proposed', proposed[seed]), ('ablation', ablation[seed])):
            report = find_report(artifact_dir, 'dev', row['method'], row['index_fingerprint'], row.get('report'))
            if report is None:
                return None, f'Missing dev report for {row["label"]} seed {seed}.'
            reports[name] = report['per_query']
        per_seed.append(reports)
    deltas = {}
    for qid in qrels:
        values = [r['proposed'][qid][proposed[seeds[0]]['method']]['metrics']['nDCG@10']
                    - r['ablation'][qid][ablation[seeds[0]]['method']]['metrics']['nDCG@10'] for r in per_seed]
        deltas[qid] = values
    mean = {qid: statistics.fmean(v) for qid, v in deltas.items()}
    consistent_up = [q for q, v in deltas.items() if all(d > 0 for d in v)]
    consistent_down = [q for q, v in deltas.items() if all(d < 0 for d in v)]
    helps = sorted(consistent_up, key=lambda q: -mean[q])[:count] or sorted(mean, key=lambda q: -mean[q])[:count]
    hurts = sorted(consistent_down, key=lambda q: mean[q])[:count] or sorted(mean, key=lambda q: mean[q])[:count]
    counts = {'queries': len(mean), 'improved': sum(v > 1e-9 for v in mean.values()),
                'worse': sum(v < -1e-9 for v in mean.values()), 'unchanged': sum(abs(v) <= 1e-9 for v in mean.values()),
                'consistently_improved_all_seeds': len(consistent_up), 'consistently_worse_all_seeds': len(consistent_down)}

    first = per_seed[0]
    method_p, method_a = proposed[seeds[0]]['method'], ablation[seeds[0]]['method']

    def rank(ranking, doc):
        return ranking.index(doc) + 1 if doc in ranking else '>100'

    lines = [f'# Dev query examples: gamma={gamma:g} vs gamma=0 (ablation)', '',
                f'Paired over seeds {seeds}. Rankings shown for seed {seeds[0]}. Delta = nDCG@10(proposed) - nDCG@10(ablation), averaged over seeds.',
                'Unjudged documents are not confirmed irrelevant; they simply have no SciFact judgment.', '',
                f"Of {counts['queries']} dev queries: {counts['improved']} improved, {counts['worse']} worse, {counts['unchanged']} unchanged (mean over seeds); "
                f"{counts['consistently_improved_all_seeds']} improved and {counts['consistently_worse_all_seeds']} worsened on every seed.", '']
    for heading, selected in (('Helps', helps), ('Hurts', hurts)):
        for qid in selected:
            ranking_p = first['proposed'][qid][method_p]['ranking']
            ranking_a = first['ablation'][qid][method_a]['ranking']
            relevant = [d for d, g in qrels[qid].items() if g > 0]
            lines += [f'## {heading}: query {qid} (mean delta nDCG@10 = {mean[qid]:+.4f}; per seed {", ".join(f"{d:+.3f}" for d in deltas[qid])})', '',
                        f'> {queries[qid]}', '', '| Relevant document | Rank, ablation | Rank, proposed |', '|---|---:|---:|']
            lines += [f'| {d}: {titles.get(d, "")[:90]} | {rank(ranking_a, d)} | {rank(ranking_p, d)} |' for d in relevant]
            lines += ['', '| # | Ablation top 5 | Proposed top 5 |', '|---:|---|---|']
            def cell(doc):
                mark = 'RELEVANT' if doc in relevant else ('judged non-relevant' if doc in qrels[qid] else 'unjudged')
                return f'{doc} [{mark}] {titles.get(doc, "")[:60]}'
            lines += [f'| {i + 1} | {cell(a)} | {cell(p)} |' for i, (a, p) in enumerate(zip(ranking_a[:5], ranking_p[:5]))]
            lines.append('')
    return counts, '\n'.join(lines)


# ---------------------------------------------------------------- main

def table(rows, columns, formats=None):
    formats = formats or {}
    head = '| ' + ' | '.join(columns) + ' |'
    rule = '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(columns))) + '|'
    def cell(row, column):
        value = row.get(column, '')
        return '' if value in ('', None) else formats.get(column, '{}').format(value)
    body = ['| ' + ' | '.join(cell(row, c) for c in columns) + ' |' for row in rows]
    return '\n'.join([head, rule, *body])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--final', default='final', help='Name of the test-split experiment.')
    parser.add_argument('--sweep', default='gamma-sweep-3seed', help='Name of the dev-split sweep experiment.')
    parser.add_argument('--gamma', type=float, default=1., help='Selected gamma for the proposed method.')
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--name', default='report')
    args = parser.parse_args()
    artifact_dir = args.artifact_dir.resolve()
    out = artifact_dir / 'analysis' / args.name
    out.mkdir(parents=True, exist_ok=True)

    baseline = load_registry(artifact_dir)['baseline']
    data_dir = resolve_resource(artifact_dir, baseline['data_dir'])
    documents, queries, test_qrels = load_dataset(data_dir, split='test')
    _, _, dev_qrels = load_dataset(data_dir, split='dev')
    titles = {str(d['_id']): d.get('title', '') for d in documents}
    entries = model_entries(artifact_dir)
    final = experiment(artifact_dir, 'test', args.final)
    sweep = experiment(artifact_dir, 'dev', args.sweep)
    rows = final['runs']
    spaces = Spaces(artifact_dir, baseline, documents)
    results = {'final': args.final, 'sweep': args.sweep, 'gamma': args.gamma}
    md = [f'# Analysis: {args.final} (test) and {args.sweep} (dev)', '',
            'Generated by `applications/analysis.py`; no model was retrained.', '']
    try:
        # ---- curves and sweep
        print('Plotting loss curves...')
        plot_curves(rows, entries, artifact_dir, out)
        sweep_table = plot_sweep(sweep, entries, artifact_dir, out)
        results['sweep_table'] = sweep_table
        md += ['## Development sweep', '', table(sweep_table, ['gamma', 'runs', 'nDCG@10', 'nDCG@10_std', 'validation_geometry'],
                                                    {'gamma': '{:g}', 'nDCG@10': '{:.4f}', 'nDCG@10_std': '{:.4f}', 'validation_geometry': '{:.2e}'}),
                '', '![gamma sweep](gamma_sweep.png)', '', '![loss components](loss_components.png)', '',
                '![train vs validation](train_vs_validation.png)', '']

        # ---- collapse diagnostics (all final runs + any contractive reference runs)
        print('Computing collapse diagnostics...')
        analysed = list(rows)
        known = {r['index_fingerprint'] for r in rows}
        for entry in entries.values():
            method = entry['config']['method']
            if 'contractive' in method and entry['index_fingerprint'] not in known:
                analysed.append({'label': f"{method} (reference, seed {entry['config']['seed']})", 'method': method,
                                    'seed': entry['config']['seed'], 'dimension': entry['dimension'],
                                    'index_fingerprint': entry['index_fingerprint']})
        collapse, spectra = [], {}
        for row in analysed:
            entry = entries.get(row['index_fingerprint'])
            raw, unit = spaces.transform(row, entry, spaces.embeddings)
            stats, spectrum = collapse_statistics(raw, unit)
            collapse.append({'run': f"{row['label']} s{row['seed']}" if row.get('seed') is not None else row['label'],
                             'dimension': row['dimension'], **stats})
            key = row['label']
            if key not in spectra and row['method'] != 'baseline':
                spectra[key] = spectrum
        plot_coordinate_std(spectra, out)
        results['collapse'] = collapse
        md += [f'## Collapse diagnostics (all {len(documents):,} documents)', '',
                'Raw = encoder output before L2 normalization; unit = normalized vectors used for retrieval. '
                'Effective dimension = participation ratio of the covariance eigenvalues of the normalized vectors. '
                'It is scale-invariant, so it cannot detect collapse by itself; use raw coordinate std and mean pairwise cosine for that.', '',
                table(collapse, ['run', 'dimension', 'raw_mean_norm', 'raw_mean_coord_std', 'raw_min_coord_std',
                                    'raw_coords_std_lt_1e-3', 'saturated_fraction_abs_gt_0.99', 'mean_pairwise_cosine', 'effective_dimension'],
                        {'raw_mean_norm': '{:.3f}', 'raw_mean_coord_std': '{:.2e}', 'raw_min_coord_std': '{:.2e}',
                        'saturated_fraction_abs_gt_0.99': '{:.4f}', 'mean_pairwise_cosine': '{:.4f}', 'effective_dimension': '{:.1f}'}),
                '', '![coordinate std](coordinate_std.png)', '']

        # ---- exact search vs HNSW on the test split
        print('Exact cosine search versus HNSW (test split)...')
        query_ids = sorted(test_qrels)
        query_vectors = spaces.queries([queries[q] for q in query_ids])
        exact_rows = []
        for row in rows:
            entry = entries.get(row['index_fingerprint'])
            _, documents_unit = spaces.transform(row, entry, spaces.embeddings)
            _, queries_unit = spaces.transform(row, entry, query_vectors)
            scores = queries_unit @ documents_unit.T
            top = np.argsort(-scores, axis=1, kind='stable')[:, :100]
            rankings = {q: [spaces.ids[i] for i in top[n]] for n, q in enumerate(query_ids)}
            metrics = [query_metrics(rankings[q], test_qrels[q]) for q in query_ids]
            exact = {m: statistics.fmean(x[m] for x in metrics) for m in METRIC_NAMES}
            report = find_report(artifact_dir, 'test', row['method'], row['index_fingerprint'], row.get('report'))
            overlap = (statistics.fmean(len(set(rankings[q]) & set(report['per_query'][q][row['method']]['ranking'])) / 100
                                        for q in query_ids) if report else float('nan'))
            exact_rows.append({'run': f"{row['label']} s{row['seed']}" if row.get('seed') is not None else row['label'],
                                'label': row['label'], 'seed': row.get('seed'), 'hnsw_nDCG@10': row['nDCG@10'], 'exact_nDCG@10': exact['nDCG@10'],
                                'hnsw_Recall@100': row['Recall@100'], 'exact_Recall@100': exact['Recall@100'],
                                'exact_MAP@100': exact['MAP@100'], 'exact_MRR@100': exact['MRR@100'],
                                'top100_overlap': overlap, 'per_query_exact_nDCG@10': [x['nDCG@10'] for x in metrics]})
        results['exact'] = [{k: v for k, v in r.items() if k != 'per_query_exact_nDCG@10'} for r in exact_rows]
        grouped = {}
        for r in exact_rows:
            grouped.setdefault(r['label'], []).append(r)
        exact_summary = [{'method': label, 'runs': len(g),
                          **{k: statistics.fmean(x[k] for x in g) for k in ('hnsw_nDCG@10', 'exact_nDCG@10', 'hnsw_Recall@100', 'exact_Recall@100', 'top100_overlap')}}
                            for label, g in grouped.items()]
        md += ['## Exact cosine search versus Chroma HNSW (test, mean over seeds)', '',
                table(exact_summary, ['method', 'runs', 'hnsw_nDCG@10', 'exact_nDCG@10', 'hnsw_Recall@100', 'exact_Recall@100', 'top100_overlap'],
                        {k: '{:.4f}' for k in ('hnsw_nDCG@10', 'exact_nDCG@10', 'hnsw_Recall@100', 'exact_Recall@100', 'top100_overlap')}), '']

        # ---- paired comparison: proposed vs ablation (and vs mse_cosine)
        print('Paired comparison...')
        proposed_label = next((r['label'] for r in rows if r.get('gamma') == args.gamma), None)
        paired = []
        for reference in [l for l in grouped if l != proposed_label and l in {r['label'] for r in rows if r.get('seed') is not None}]:
            p = {r['seed']: r for r in grouped.get(proposed_label, [])}
            a = {r['seed']: r for r in grouped[reference]}
            seeds = sorted(set(p) & set(a))
            if not seeds:
                continue
            per_seed = [p[s]['exact_nDCG@10'] - a[s]['exact_nDCG@10'] for s in seeds]
            per_query = np.mean([np.array(p[s]['per_query_exact_nDCG@10']) - np.array(a[s]['per_query_exact_nDCG@10']) for s in seeds], axis=0)
            mean, low, high = bootstrap_mean_ci(per_query)
            paired.append({'comparison': f'{proposed_label} minus {reference}', 'seeds': len(seeds),
                            'per_seed_deltas': ', '.join(f'{d:+.4f}' for d in per_seed),
                            'mean_delta': mean, 'ci95_low': low, 'ci95_high': high,
                            'queries_better': int((per_query > 1e-9).sum()), 'queries_worse': int((per_query < -1e-9).sum()),
                            'queries_tied': int((np.abs(per_query) <= 1e-9).sum())})
        results['paired'] = paired
        md += ['## Paired comparison on test nDCG@10 (exact search)', '',
                'Per-seed deltas pair runs with the same initialization seed. The confidence interval bootstraps the ' + str(len(test_qrels)) + ' test queries '
                '(10,000 resamples) using each query\'s delta averaged over seeds; it captures query sampling, not seed variability.', '',
                table(paired, ['comparison', 'seeds', 'per_seed_deltas', 'mean_delta', 'ci95_low', 'ci95_high', 'queries_better', 'queries_worse', 'queries_tied'],
                        {'mean_delta': '{:+.4f}', 'ci95_low': '{:+.4f}', 'ci95_high': '{:+.4f}'}), '']

        # ---- storage and parameters
        storage = []
        for label in dict.fromkeys(r['label'] for r in rows):
            row = next(r for r in rows if r['label'] == label)
            storage.append({'method': label, 'dimension': row['dimension'],
                            'bytes_per_document': row['dimension'] * FLOAT_BYTES,
                            'corpus_MB': row['dimension'] * FLOAT_BYTES * len(documents) / 1e6,
                            'encoder_parameters': row.get('parameter_count') or 0,
                            'mean_training_seconds': statistics.fmean(r['training_seconds'] for r in rows if r['label'] == label) if row.get('training_seconds') else ''})
        results['storage'] = storage
        md += ['## Storage and size (raw float32 vectors, excluding index overhead)', '',
                table(storage, ['method', 'dimension', 'bytes_per_document', 'corpus_MB', 'encoder_parameters', 'mean_training_seconds'],
                        {'corpus_MB': '{:.2f}', 'mean_training_seconds': '{:.1f}'}), '']

        # ---- query examples on dev
        print('Selecting dev query examples...')
        counts, examples = query_examples(sweep, args.gamma, artifact_dir, queries, dev_qrels, titles)
        (out / 'query_examples.md').write_text(examples, encoding='utf-8')
        results['dev_query_counts'] = counts
        md += ['## Dev query-level effect', '', str(counts), '', 'See [query_examples.md](query_examples.md).', '']
    finally:
        spaces.close()
    (out / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    write_json(out / 'summary.json', results)
    print('\n'.join(md))
    print('\nWrote', out)


if __name__ == '__main__':
    main()
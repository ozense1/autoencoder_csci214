"""Run from the project root: streamlit run applications/app.py"""
import json
import os
from pathlib import Path
import shlex
import sys

# Support both `streamlit run applications/app.py` and Streamlit's AppTest.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from config import ARTIFACT_DIR, DATA_DIR, DEFAULT_MODEL, FIXED_METHODS, METHODS
from data_utils import fingerprint, load_dataset, read_json, resolve_resource
from embeddings import Embedder, embedding_spec
from metrics import query_metrics
from retrieval import Retriever, evaluate, query_embedding
from storage import load_registry, readiness

st.set_page_config(page_title='SciFact Retrieval Lab', page_icon='🔎', layout='wide')


@st.cache_data(show_spinner=False)
def dataset(path, corpus_mtime, queries_mtime, qrels_mtime):
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def current_spec(model_path, model_size, model_mtime, data_dir, corpus_mtime):
    return embedding_spec(model_path, data_dir)


@st.cache_resource(show_spinner='Loading the local embedding model…')
def embedding_model(path, model_fingerprint, threads, gpu_layers):
    return Embedder(path, threads=threads, gpu_layers=gpu_layers)


@st.cache_resource(show_spinner=False)
def retriever(artifact_dir):
    return Retriever(artifact_dir)


def command(method, model_path, data_dir, artifact_dir):
    parts = [sys.executable, str(Path(__file__).with_name('train_autoencoder.py')),
                'prepare' if method in FIXED_METHODS else 'train']
    if method not in FIXED_METHODS:
        parts += ['--variant', method]
    parts += ['--model-path', str(model_path), '--data-dir', str(data_dir), '--artifact-dir', str(artifact_dir)]
    return shlex.join(parts)


def show_results(result, entries, top_k, judgments):
    embed_time = result['embedding_timing']
    st.caption(f"Query embedding {'cache read' if embed_time['cached'] else 'inference'}: {embed_time['seconds'] * 1000:.1f} ms · encoded once for all methods")
    if embed_time['truncated']:
        st.warning('The query was truncated to the model’s 2,048-token context.')
    metric_rows = []
    for method, outcome in result['methods'].items():
        timings = outcome['timings']
        row = {'Method': METHODS[method], 'Dimensions': entries[method]['dimension'],
               'Encoder ms': timings['encoder_seconds'] * 1000, 'Search ms': timings['search_seconds'] * 1000}
        if judgments is not None:
            metrics = query_metrics([doc['id'] for doc in outcome['documents']], judgments)
            row.update({{'MAP@100': 'AP@100', 'MRR@100': 'RR@100'}.get(key, key): value for key, value in metrics.items()})
        metric_rows.append(row)
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width='stretch')
    st.caption('Cosine scores from different representation spaces are not calibrated against each other.')
    if judgments is None:
        st.caption('No relevance judgments are available for this free-text query. Cosine scores are similarities, not relevance probabilities.')
    else:
        st.caption('Metrics use the first 100 retrieved documents, independently of the displayed top-k. Unjudged documents contribute no relevance gain.')
    methods = list(result['methods'])
    # At most three readable result columns per row, even when comparing all five.
    for start in range(0, len(methods), 3):
        columns = st.columns(min(3, len(methods) - start))
        for column, method in zip(columns, methods[start:start + 3]):
            with column:
                st.subheader(METHODS[method].replace('Embedding + AE: ', 'AE: '))
                for rank, document in enumerate(result['methods'][method]['documents'][:top_k], 1):
                    known = judgments is not None and judgments.get(document['id'], 0) > 0
                    with st.container(border=True):
                        if known:
                            st.badge('Judged relevant', icon=':material/check_circle:', color='green')
                        elif judgments is not None:
                            st.caption('Unjudged' if document['id'] not in judgments else 'Judged nonrelevant')
                        st.markdown(f"**{rank}. {document['title'] or 'Untitled document'}**")
                        st.caption(f"Document {document['id']} · cosine {document['cosine']:.4f}")
                        st.write(document['text'][:240] + ('…' if len(document['text']) > 240 else ''))
                        with st.expander('Read abstract'):
                            st.write(document['text'])


def main():
    artifact_dir = Path(os.getenv('CAE_ARTIFACT_DIR', str(ARTIFACT_DIR))).expanduser().resolve()
    registry = load_registry(artifact_dir)
    baseline = registry.get('baseline')
    stored_model = resolve_resource(artifact_dir, baseline['model_path']) if baseline else DEFAULT_MODEL
    stored_data = resolve_resource(artifact_dir, baseline['data_dir']) if baseline else DATA_DIR
    model_path = Path(os.getenv('CAE_MODEL_PATH', str(stored_model))).expanduser().resolve()
    data_dir = Path(os.getenv('CAE_DATA_DIR', str(stored_data))).expanduser().resolve()
    st.title('SciFact Retrieval Lab')
    st.caption('Compare original embeddings with autoencoders trained using reconstruction, cosine, and contractive loss.')
    try:
        documents, queries, qrels = dataset(str(data_dir), (data_dir / 'corpus.jsonl').stat().st_mtime_ns,
                                            (data_dir / 'queries.jsonl').stat().st_mtime_ns,
                                            (data_dir / 'qrels' / 'test.tsv').stat().st_mtime_ns)
    except (OSError, ValueError) as exc:
        st.error(f'Cannot load SciFact: {exc}')
        st.info('See applications/README.md for the dataset-copy command.')
        return
    available = readiness(artifact_dir)
    entries = {key: value['entry'] for key, value in available.items() if value['ready']}
    with st.sidebar:
        st.header('Retrieval methods')
        for key in METHODS:
            st.markdown(f"**{METHODS[key].replace('Embedding + AE: ', 'AE: ')}**")
            st.caption(f"Ready · {entries[key]['dimension']} dimensions" if key in entries else 'Needs preparation')
        st.button('Refresh available methods')
        with st.expander('Local setup'):
            st.caption(f'Model: {model_path.name}')
            st.caption(f'Data: {data_dir}')
            st.caption(f'Artifacts: {artifact_dir}')
            st.code(command('baseline', model_path, data_dir, artifact_dir), language='bash')
        st.caption('All autoencoders use document vectors only. Relevance labels are reserved for evaluation.')
    if 'baseline' not in entries:
        st.info('Start with embedding-only retrieval. Prepare the document embeddings and Chroma index once:')
        st.code(command('baseline', model_path, data_dir, artifact_dir), language='bash')
        for value in available.values():
            if value.get('reason'):
                st.warning(value['reason'])
        return
    try:
        stat = model_path.stat()
        _, actual_fingerprint = current_spec(str(model_path), stat.st_size, stat.st_mtime_ns, str(data_dir),
                                                (data_dir / 'corpus.jsonl').stat().st_mtime_ns)
        if actual_fingerprint != baseline['base_fingerprint']:
            raise ValueError('The model, corpus, or embedding runtime changed. Prepare matching indexes before searching.')
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        st.code(command('baseline', model_path, data_dir, artifact_dir), language='bash')
        return
    runtime = baseline['runtime']
    engine = lambda: embedding_model(str(model_path), actual_fingerprint, runtime['threads'], runtime['gpu_layers'])
    counts = st.columns(3)
    counts[0].metric('Documents', f'{len(documents):,}')
    counts[1].metric('Original dimensions', baseline['dimension'])
    counts[2].metric('Held-out test queries', len(qrels))
    search_tab, evaluation_tab, training_tab = st.tabs(['Search & compare', 'Benchmark evaluation', 'Training & setup'])
    with search_tab:
        view = st.radio('View', ['Single method', 'Compare methods'], horizontal=True, key='view')
        if view == 'Single method':
            selected = [st.selectbox('Method', list(METHODS), format_func=METHODS.get, key='method')]
        else:
            defaults = ['baseline'] + [key for key in METHODS if key != 'baseline' and key in entries][:1]
            selected = st.multiselect('Methods to compare', list(METHODS), default=defaults, format_func=lambda key: METHODS[key].replace('Embedding + AE: ', ''), key='compare_methods')
        source = st.radio('Query source', ['SciFact test query', 'Free text'], horizontal=True, key='query_source')
        query_id = None
        if source == 'SciFact test query':
            query_id = st.selectbox('Test query', sorted(qrels, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k)),
                                    format_func=lambda key: f'{key} · {queries[key]}', key='query_id')
            query = queries[query_id]
            st.write(query)
        else:
            query = st.text_area('Your query', placeholder='For example: Does vitamin D improve bone health?', key='query_text').strip()
        top_k = st.slider('Documents shown per method', 1, 20, 5, key='top_k')
        missing = [method for method in selected if method not in entries]
        for method in missing:
            st.warning(f'{METHODS[method]} is not ready.')
            st.code(command(method, model_path, data_dir, artifact_dir), language='bash')
        signature = fingerprint({'query': query, 'id': query_id, 'methods': selected,
                                    'indexes': {m: entries[m]['index_fingerprint'] for m in selected if m in entries}})
        if st.button('Search', type='primary', disabled=bool(missing) or not selected or not query, key='search'):
            try:
                with st.spinner('Retrieving documents…'):
                    vector, timing = query_embedding(query, baseline, artifact_dir, engine)
                    outcomes = {}
                    service = retriever(str(artifact_dir))
                    for method in selected:
                        rows, timings = service.search(vector, method, entries[method], top_k=100)
                        outcomes[method] = {'documents': rows, 'timings': timings}
                    st.session_state['search_result'] = {'signature': signature, 'embedding_timing': timing, 'methods': outcomes}
            except Exception as exc:
                st.error(f'Search could not complete: {exc}')
        result = st.session_state.get('search_result')
        if result and result['signature'] == signature:
            show_results(result, entries, top_k, qrels[query_id] if query_id is not None else None)
    with evaluation_tab:
        st.subheader('Evaluate on all SciFact test queries')
        st.write('Compare nDCG@10, Recall@10, Recall@100, MAP@100 and MRR@100 using the same corpus and 100 retrieved candidates per query.')
        selected_eval = st.multiselect('Methods to evaluate', list(entries), default=list(entries), format_func=lambda key: METHODS[key].replace('Embedding + AE: ', ''), key='evaluation_methods')
        st.caption('This is a transductive document-compression experiment: the document corpus is used without labels for AE training. Test queries and judgments are never training inputs.')
        if st.button('Run benchmark', disabled=not selected_eval, key='evaluate'):
            bar = st.progress(0., text='Preparing queries…')
            try:
                report = evaluate(selected_eval, entries, baseline, artifact_dir, queries, qrels, engine,
                                    progress=lambda done, total: bar.progress(done / total, text=f'Evaluated {done}/{total} queries'))
                st.session_state['evaluation_report'] = report
            except Exception as exc:
                st.error(f'Evaluation could not complete: {exc}')
            finally:
                bar.empty()
        report = st.session_state.get('evaluation_report')
        if report is None:
            # Show the newest compatible saved result without automatically running a benchmark.
            files = sorted((artifact_dir / 'evaluations').glob('*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
            for path in files:
                candidate = read_json(path)
                if (candidate['base_fingerprint'] == baseline['base_fingerprint']
                    and candidate['judgments_fingerprint'] == fingerprint(qrels)
                    and candidate['queries_fingerprint'] == fingerprint(queries)
                    and all(m in entries and values['index_fingerprint'] == entries[m]['index_fingerprint'] for m, values in candidate['methods'].items())):
                    report = candidate
                    break
        if report and all(m in entries and values['index_fingerprint'] == entries[m]['index_fingerprint'] for m, values in report['methods'].items()) and report['base_fingerprint'] == baseline['base_fingerprint'] and report['judgments_fingerprint'] == fingerprint(qrels) and report['queries_fingerprint'] == fingerprint(queries):
            st.caption(f"Saved evaluation · {report['query_count']} test queries. Latencies include local caching and are illustrative, not a controlled speed benchmark.")
            table = [{'Method': METHODS[m], 'Dimensions': value['dimension'], **value['metrics'],
                      'Mean search ms': value['mean_search_seconds'] * 1000} for m, value in report['methods'].items()]
            st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
            st.download_button('Download evaluation JSON', json.dumps(report, indent=2), file_name='scifact_evaluation.json', mime='application/json')
    with training_tab:
        st.subheader('Train the autoencoder variants')
        st.write('Training runs from the command line and reuses the cached embeddings. The app reads completed checkpoints and indexes.')
        all_command = command('mse', model_path, data_dir, artifact_dir).replace('--variant mse ', '--variant all ')
        st.code(all_command, language='bash')
        st.caption('Defaults: 768 → 128 → 768 · tanh encoder · tied linear decoder · α=1, β=0.1, λ=0.01 where applicable. These are example settings, not tuned recommendations.')
        st.latex(r'L=\alpha\,\mathrm{MSE}(x,\hat{x})+\beta[1-\cos(x,\hat{x})]+\lambda\|\partial z/\partial x\|_F^2+\gamma\,\tfrac{1}{B(B-1)}\sum_{i\neq j}(S^x_{ij}-S^z_{ij})^2')
        trained = [key for key in entries if 'checkpoint' in entries[key]]
        if trained:
            method = st.selectbox('Training run', trained, format_func=METHODS.get, key='training_method')
            entry = entries[method]
            history = pd.DataFrame(read_json(artifact_dir / entry['history'])).set_index('epoch')
            st.line_chart(history[['train_total', 'validation_total']], x_label='Epoch', y_label='Weighted objective')
            components = [c for c in ('validation_mse', 'validation_cosine', 'validation_contractive', 'validation_geometry') if c in history]
            for component in components:
                # Separate charts: the terms differ by orders of magnitude.
                st.line_chart(history[[component]], x_label='Epoch', y_label=component.replace('validation_', 'Validation '))
            if 'latent_stats' in entry:
                st.caption('Collapse diagnostics (raw z, all documents)')
                st.json(entry['latent_stats'])
            st.caption(f"Best epoch {entry['best_epoch']} of {entry['epochs_completed']}; checkpoint selected using unlabeled document validation loss.")
            with st.expander('Checkpoint settings'):
                st.json(entry['config'])
        st.info('A smaller Jacobian penalty or reconstruction loss does not guarantee better retrieval. Compare the held-out relevance metrics.')
        if baseline['truncated_documents']:
            st.caption(f"{baseline['truncated_documents']} documents were truncated at 2,048 tokens.")


if __name__ == '__main__':
    main()

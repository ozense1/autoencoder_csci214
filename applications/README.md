# SciFact Retrieval Lab

A local Streamlit application comparing EmbeddingGemma retrieval with four autoencoder objectives. It uses the copied SciFact corpus, PyTorch, and a persistent ChromaDB database. Autoencoder training uses **individual document embeddings only**, never query–document pairs or relevance judgments.

## Quick start

Run these commands from the project root:

```bash
# If the dependencies are not already installed:
python -m pip install -r applications/requirements.txt

# Start with original 768-dimensional EmbeddingGemma retrieval:
python applications/train_autoencoder.py prepare

# Train all four 128-dimensional autoencoders and build their indexes:
python applications/train_autoencoder.py train --variant all

# Start the interface:
streamlit run applications/app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Prepared artifacts are reused. In the supplied workspace, the corpus, indexes and trained checkpoints have already been prepared; you can start the interface directly.

The embedding model is bundled inside the application:

```text
applications/models/embeddinggemma-300m-Q4_0.gguf
```

No external model directory, model download, or Chroma server is needed. `models/manifest.json` records the bundled GGUF checksum. ChromaDB runs in-process and persists to `applications/artifacts/chroma`.

## Interface

The **Search & compare** tab supports a single method or selected methods side by side. Enter free text or choose a SciFact test query. Results include rank, document ID, title, cosine similarity, and an expandable abstract. Known relevant documents are highlighted for benchmark queries; missing judgments are labeled **unjudged**, not assumed to be known negatives.

| Option | Objective | Retrieval dimensions |
|---|---|---:|
| Embedding only | No autoencoder | 768 |
| Embedding + AE: MSE | MSE | 128 by default |
| Embedding + AE: MSE + Cosine | MSE + cosine reconstruction | 128 by default |
| Embedding + AE: MSE + Cosine + Contractive | All three terms | 128 by default |
| Embedding + AE: MSE + Contractive | MSE + contractive penalty | 128 by default |

The **Benchmark evaluation** tab runs selected methods over all **300 test queries**, fetching 100 candidates per query. It reports nDCG@10, Recall@10, Recall@100, MAP@100, and MRR@100 and lets you download the detailed results. Changing the number of displayed documents does not change evaluation cutoffs. Saved compatible evaluations are shown on startup.

The **Training & setup** tab shows training commands, loss curves, selected checkpoint epochs, and configuration. Training remains a CLI operation so UI reruns cannot restart it. Missing variants show the exact preparation command and are unavailable for search until their checkpoint and index are complete.

Cosine values are not relevance probabilities, and scores in different latent spaces are not directly calibrated. Query encoding/cache time, latent encoding time, and index search time are shown separately; timings are illustrative local measurements, not controlled speed benchmarks.

## Dataset and embedding preparation

`data/scifact` contains an independent copy of:

```text
data/scifact/
  corpus.jsonl        # 5,183 documents
  queries.jsonl       # 1,109 queries
  qrels/train.tsv     # 809 queries, 919 judgments
  qrels/test.tsv      # 300 queries, 339 judgments
```

The dataset is bundled in `applications/data/scifact`. `data/copy_manifest.json` records SHA-256 checksums and byte counts verified when the data was copied. No desktop dataset directory is needed to run the application.

Queries use `task: search result | query: {query}`. Documents use `title: {title or none} | text: {abstract}`. Both are embedded with the frozen model and L2-normalized. The full 768-dimensional output is retained: this experiment does not use the model's built-in Matryoshka truncation.

The installed runtime supports single-text encoding for this GGUF; a list of strings in one `embed` call fails. The wrapper therefore serializes one-text calls. Documents are truncated to the first 2,048 token IDs including tokenizer-added special tokens, following the runtime's high-level embedding API. Truncation counts are recorded and query truncation is displayed in the UI.

Preparation saves progress every 32 documents. Rerun the same `prepare` command after interruption; completed chunks are reused. Finished vectors are indexed in an explicit cosine-space Chroma collection without Chroma's default embedding function or any automatic model download.

Model content, corpus content, formatting, context size, normalization and binding version identify embedding caches. Each trained checkpoint gets a separate fingerprinted collection. Model, dataset or checkpoint mismatches are rejected instead of silently mixing vector spaces. Changing training settings creates a separate run and activates its matching index; older artifacts remain on disk.

## Training configuration

```bash
# Train one objective:
python applications/train_autoencoder.py train --variant mse_cosine_contractive

# Smaller latent space and different regularization strength:
python applications/train_autoencoder.py train --variant all \
  --latent-dim 64 --lambda 0.001 --epochs 100

# Explicit model and dataset locations:
python applications/train_autoencoder.py prepare \
  --model-path /absolute/path/embeddinggemma-300m-Q4_0.gguf \
  --data-dir applications/data/scifact

# Optional llama.cpp acceleration if your installation supports it:
python applications/train_autoencoder.py prepare --gpu-layers -1

# Full test evaluation without Streamlit:
python applications/evaluate.py
```

Additional options are listed by `python applications/train_autoencoder.py --help`. The defaults are:

| Setting | Default |
|---|---|
| Architecture | 768 → 128 → 768 |
| Encoder | Linear + tanh |
| Decoder | Transposed encoder weights + independent bias; linear output |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 64 |
| Maximum epochs | 100 |
| Early stopping | 10 epochs without validation improvement of at least 1e-8 |
| Split | Deterministic 90/10 split of document vectors |
| Seed | 7 |
| Training device | CPU |
| Threads | 4 |
| Embedding layer offload | 0; optional `--gpu-layers -1` |
| α, β, λ | 1.0, 0.1, 0.01, with unused terms disabled |

Each variant starts with the same weights, split and minibatch order. The checkpoint with the lowest weighted validation objective is selected. The objective weights are examples, not tuned recommendations. Test relevance judgments do not affect training, early stopping, or checkpoint selection.

For a tanh encoder, the exact penalty is:

```text
z = tanh(Wx + b)
x_hat = Wᵀz + decoder_bias

J[i,j] = (1 - z[i]²) W[i,j]
L_contractive = mean_over_batch Σ_i [(1-z[i]²)² Σ_j W[i,j]²]

L = α MSE(x, x_hat)
  + β mean_over_batch [1 - cosine(x, x_hat)]
  + λ L_contractive
```

MSE averages over input coordinates and batch. The contractive term sums Jacobian entries before averaging over the batch. Its analytic implementation preserves parameter gradients but avoids constructing a `[batch, 128, 768]` Jacobian. Unit tests compare both its value and parameter gradients with exact autograd.

The penalty applies to **raw z**, before L2 normalization for retrieval. The trained encoder is applied to both query and document vectors. The decoder is unused during retrieval. Tied weights prevent freely shrinking the encoder and independently amplifying decoder weights; they do not guarantee useful geometry or prevent every form of collapse.

Because the target document corpus supplies unlabeled training vectors, this is a **transductive document-compression experiment**. The held-out test queries and qrels are not training inputs. Lower loss, smaller vectors, or smaller Jacobians do not imply better retrieval. Test the relevance metrics, and use separate development data for future hyperparameter selection.

## Metrics and evaluation artifacts

- nDCG@10 uses gain `2^relevance - 1` and logarithmic rank discounts, normalized by the ideal top-ten ranking. SciFact judgments are binary.
- Recall@k divides retrieved relevant documents by all known relevant documents for that query.
- AP@100 sums precision at relevant hits within rank 100 and divides by the total known relevant count. MAP@100 averages it over all test queries.
- RR@100 is the reciprocal rank of the first relevant hit within 100, or zero. MRR@100 averages it over queries.
- Queries with no relevant documents return zero metrics; the supplied 300 test queries have positive judgments.

Single-query tables use AP and RR labels; aggregate benchmark tables use MAP and MRR. Unjudged documents contribute zero relevance gain for scoring, although the UI keeps the distinction from explicitly judged negatives. All indexes use identical HNSW settings (`space=cosine`, `ef_construction=200`, `ef_search=200`, `max_neighbors=16`, four threads). These are approximate-neighbor retrieval results, so index approximation is part of the measured pipeline.

Reports in `artifacts/evaluations` include per-query top-100 rankings, relevance metrics, index fingerprints, dimensions, query-cache counts and timing information. No fabricated benchmark results are included.

## Files and portability

```text
applications/
  app.py                   Streamlit interface
  train_autoencoder.py     Preparation and training CLI
  evaluate.py              Full test-set evaluation CLI
  autoencoder.py           Tied model and exact analytic penalty
  training.py              Checkpoint selection and latent indexing
  prepare.py               Resumable document embedding cache
  embeddings.py            Local GGUF inference and text formatting
  storage.py               Persistent Chroma collections and readiness
  retrieval.py             Query caches, vector search and evaluation
  metrics.py               Retrieval metrics
  models/                  Bundled EmbeddingGemma GGUF and checksum manifest
  data/                    Copied SciFact data and copy checksums
  artifacts/               Embeddings, checkpoints, indexes, histories, reports
  tests/                   Numerical, pipeline and Streamlit tests
```

**To move the app, copy the entire `applications/` directory**, including `models/`, `data/`, and `artifacts/`. Keep Streamlit and training stopped while copying the Chroma database. The existing five indexes and trained checkpoints can be reused without re-embedding or retraining:

```bash
cd /new/location/applications
python -m pip install -r requirements.txt
streamlit run app.py
```

Defaults are anchored to the location of the Python files, not the working directory. The registry stores model and dataset paths relative to its artifact directory (`../models/...` and `../data/scifact`); checkpoints and embedding caches are also referenced relative to that directory. Moving the directory does not change content fingerprints.

`artifacts/` is ignored by Git, so copying only tracked source files will omit the prepared indexes and checkpoints. Include it explicitly for a ready-to-use transfer. If omitted, use the preparation and training commands to rebuild it from the bundled model and dataset.

Optional `CAE_MODEL_PATH`, `CAE_DATA_DIR`, and `CAE_ARTIFACT_DIR` overrides remain available. External overrides must be moved separately or updated on the destination. They are not needed for the bundled configuration. For example:

```bash
CAE_MODEL_PATH=/absolute/path/embeddinggemma-300m-Q4_0.gguf \
  streamlit run applications/app.py
```

CLI path overrides are `--model-path`, `--data-dir`, and `--artifact-dir`. No path edits or preparation rerun are needed when moving the complete bundled application. If you choose different model or corpus contents, prepare new embeddings and train matching autoencoders.

Dependencies were tested with Python 3.14.3, Streamlit 1.56.0, ChromaDB 1.5.9, NumPy 2.4.4, pandas 3.0.2, PyTorch 2.12.0, and the locally installed llama-cpp-python 0.3.21. The binding's native llama.cpp build must support the `gemma-embedding` GGUF architecture. Custom builds can differ despite sharing a Python package version. If a fresh installation reports an unknown architecture or cannot encode a single text, install/rebuild a compatible llama.cpp binding; the app does not silently substitute another model. No GPU is required. GPU offload requires an appropriate native build.

Run checks with:

```bash
python -m pytest applications/tests -q
```

The tests cover analytic parameter gradients, all four losses, interrupted cache resumption, fingerprints, normalized vectors, copied data, Chroma persistence and isolation, checkpoint reloads, hand-calculated metrics, and the Streamlit missing-artifact/search/comparison/evaluation states. UI unit tests use tiny deterministic data; the completed application is also validated against the real model and full corpus.

## Recorded validation

See [VALIDATION.md](VALIDATION.md) for the measured 300-query results and completed checks. These default settings did not outperform the embedding-only baseline.

## References

- [EmbeddingGemma model card and retrieval prompts](https://huggingface.co/google/embeddinggemma-300m/blob/main/README.md)
- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [Streamlit application testing](https://docs.streamlit.io/develop/api-reference/app-testing)
- [BEIR benchmark](https://arxiv.org/abs/2104.08663) and [official dataset documentation](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [Contractive Auto-Encoders, Rifai et al., ICML 2011](https://icml.cc/2011/papers/455_icmlpaper.pdf)
- [TREC average precision at cutoffs](https://github.com/usnistgov/trec_eval/blob/main/m_map_cut.c)

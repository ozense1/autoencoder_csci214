# Validation and first SciFact run

Completed on 21 September 2026 using the local EmbeddingGemma Q4_0 file and all 5,183 SciFact documents. Autoencoders used the planned defaults: 128 latent dimensions, seed 7, α=1, β=0.1 and λ=0.01 where applicable. This is one configuration and one seed, not a claim of optimal performance.

## Checks

- Copied dataset checksums match the source; 300 held-out test queries were evaluated.
- All 19 automated tests pass, including exact parameter-gradient comparisons and Streamlit interaction tests.
- The real model returns finite normalized 768-dimensional embeddings for queries, documents and an overlength query; the latter correctly reports truncation.
- Every collection contains 5,183 documents. All four checkpoints reload successfully and are matched to their indexes.
- A real Streamlit AppTest session searched the baseline and all five comparison methods without errors.
- A headless Chromium session verified baseline search, comparison and the saved benchmark display. Browser previews are in `artifacts/*_preview.png`.
- The local Streamlit HTTP health check returned `ok`.

## Measured test results

| Method | Dimensions | nDCG@10 | Recall@10 | Recall@100 | MAP@100 | MRR@100 |
|---|---:|---:|---:|---:|---:|---:|
| Embedding only | 768 | 0.7817 | 0.9068 | 0.9833 | 0.7403 | 0.7525 |
| MSE | 128 | 0.7424 | 0.8936 | 0.9767 | 0.6936 | 0.7050 |
| MSE + Cosine | 128 | 0.7487 | 0.8952 | 0.9833 | 0.7008 | 0.7132 |
| MSE + Cosine + Contractive | 128 | 0.7002 | 0.8381 | 0.9637 | 0.6582 | 0.6698 |
| MSE + Contractive | 128 | 0.0200 | 0.0351 | 0.1238 | 0.0187 | 0.0196 |

The original embeddings perform best in this run. MSE + Cosine preserves much of the retrieval quality at one-sixth of the original dimensions, but still scores below the baseline. The selected MSE + Contractive checkpoint performs poorly. Its average raw latent-coordinate standard deviation across documents is approximately 3.95×10⁻⁵, indicating almost constant codes for this configuration. No hyperparameters were changed based on these test results.

Contractive losses and training objectives have different numerical scales. The example λ is not a tuned recommendation; use separate development data for any future relevance-based tuning.

## Selected epochs

| Method | Best epoch | Epochs completed |
|---|---:|---:|
| MSE | 48 | 58 |
| MSE + Cosine | 48 | 58 |
| MSE + Cosine + Contractive | 29 | 39 |
| MSE + Contractive | 13 | 23 |

Detailed per-query rankings and artifact fingerprints: [evaluation JSON](artifacts/evaluations/246b4c08dddd466caea09d2bd9c2e8f744688208150c8ec0071bded0e02a0c09.json).

Search latency numbers in that JSON are local pipeline observations influenced by caching, method order and concurrent validation work. They are not a controlled speed comparison.

## Portability refactor

The embedding model is bundled at `models/embeddinggemma-300m-Q4_0.gguf`; SciFact remains bundled under `data/scifact`. Resource paths in the registry are relative to the artifact directory. No source code, configuration JSON, or application documentation references the original user home directory.

The entire application was copied into a temporary directory and exercised from a different working directory, with no path overrides. All five retrieval methods passed using the copied model, the relocated Streamlit app completed a search, the training CLI reused all four existing checkpoints, and the evaluation CLI completed all 300 test queries. The temporary copy was then removed. See [portability validation](artifacts/portability_validation.json).

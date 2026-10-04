# Dev query examples: gamma=1 vs gamma=0 (ablation)

Paired over seeds [1, 2, 3]. Rankings shown for seed 1. Delta = nDCG@10(proposed) - nDCG@10(ablation), averaged over seeds.
Unjudged documents are not confirmed irrelevant; they simply have no SciFact judgment.

Of 809 dev queries: 115 improved, 62 worse, 632 unchanged (mean over seeds); 24 improved and 10 worsened on every seed.

## Helps: query 1308 (mean delta nDCG@10 = +0.4941; per seed +0.613, +0.500, +0.369)

> Tonic signaling from the scFv is amplified by MyD88/CD40.

| Relevant document | Rank, ablation | Rank, proposed |
|---|---:|---:|
| 18231807: 4-1BB Costimulation Ameliorates T Cell Exhaustion Induced by Tonic Signaling of Chimeric A | 5 | 1 |

| # | Ablation top 5 | Proposed top 5 |
|---:|---|---|
| 1 | 25747721 [unjudged] Concomitant induction of CD4+ and CD8+ T cell responses in v | 18231807 [RELEVANT] 4-1BB Costimulation Ameliorates T Cell Exhaustion Induced by |
| 2 | 33397197 [unjudged] Re-engineered CD40 receptor enables potent pharmacological a | 27567994 [unjudged] CD169-positive macrophages dominate antitumor immunity by cr |
| 3 | 18834078 [unjudged] Secondary T cell–T cell synaptic interactions drive the diff | 25747721 [unjudged] Concomitant induction of CD4+ and CD8+ T cell responses in v |
| 4 | 40473317 [unjudged] Temporal segregation of 4-1BB versus CD28-mediated costimula | 33397197 [unjudged] Re-engineered CD40 receptor enables potent pharmacological a |
| 5 | 18231807 [RELEVANT] 4-1BB Costimulation Ameliorates T Cell Exhaustion Induced by | 18237384 [unjudged] Inducing Tumor Immunity through the Selective Engagement of  |

## Helps: query 20 (mean delta nDCG@10 = +0.4127; per seed +0.500, +0.369, +0.369)

> 7-day-old adult Caenorhabditis elegans exhibit approximately 75% of the learning capacity seen in 1-day old adults.

| Relevant document | Rank, ablation | Rank, proposed |
|---|---:|---:|
| 3202143: Insulin Signaling and Dietary Restriction Differentially Influence the Decline of Learning | 3 | 1 |

| # | Ablation top 5 | Proposed top 5 |
|---:|---|---|
| 1 | 36178047 [unjudged] Decrease of learning capacity in offspring with increasing p | 3202143 [RELEVANT] Insulin Signaling and Dietary Restriction Differentially Inf |
| 2 | 13072113 [unjudged] Neurite sprouting and synapse deterioration in the aging Cae | 36178047 [unjudged] Decrease of learning capacity in offspring with increasing p |
| 3 | 3202143 [RELEVANT] Insulin Signaling and Dietary Restriction Differentially Inf | 13072113 [unjudged] Neurite sprouting and synapse deterioration in the aging Cae |
| 4 | 16627684 [unjudged] Hmga2 Promotes Neural Stem Cell Self-Renewal in Young but No | 16627684 [unjudged] Hmga2 Promotes Neural Stem Cell Self-Renewal in Young but No |
| 5 | 5395426 [unjudged] Excitotoxic brain injury in adult zebrafish stimulates neuro | 1733337 [unjudged] Age-related changes in kynurenic acid production in rat brai |

## Hurts: query 766 (mean delta nDCG@10 = -0.3691; per seed -0.369, -0.369, -0.369)

> Medications to treat obesity have unwanted side effects.

| Relevant document | Rank, ablation | Rank, proposed |
|---|---:|---:|
| 7552215: Long term pharmacotherapy for obesity and overweight: updated meta-analysis. | 1 | 2 |

| # | Ablation top 5 | Proposed top 5 |
|---:|---|---|
| 1 | 7552215 [RELEVANT] Long term pharmacotherapy for obesity and overweight: update | 5839365 [unjudged] Anti-obesity drugs: past, present and future |
| 2 | 5839365 [unjudged] Anti-obesity drugs: past, present and future | 7552215 [RELEVANT] Long term pharmacotherapy for obesity and overweight: update |
| 3 | 1428830 [unjudged] The atypical antipsychotic olanzapine causes weight gain by  | 1428830 [unjudged] The atypical antipsychotic olanzapine causes weight gain by  |
| 4 | 38752049 [unjudged] A chart review of cyproheptadine for stimulant-induced weigh | 5268462 [unjudged] Obesity and Its Metabolic Complications: The Role of Adipoki |
| 5 | 28845338 [unjudged] Increased lean body mass as an early indicator of olanzapine | 38752049 [unjudged] A chart review of cyproheptadine for stimulant-induced weigh |

## Hurts: query 1068 (mean delta nDCG@10 = -0.3691; per seed -0.369, -0.369, -0.369)

> ScPif1p has reduced binding ability to G-rich ssDNA compared to non-G-rich ssDNA.

| Relevant document | Rank, ablation | Rank, proposed |
|---|---:|---:|
| 4429668: Pif1 family helicases suppress genome instability at G-quadruplex motifs | 1 | 2 |

| # | Ablation top 5 | Proposed top 5 |
|---:|---|---|
| 1 | 4429668 [RELEVANT] Pif1 family helicases suppress genome instability at G-quadr | 25787749 [unjudged] G-quadruplexes significantly stimulate Pif1 helicase-catalyz |
| 2 | 25787749 [unjudged] G-quadruplexes significantly stimulate Pif1 helicase-catalyz | 4429668 [RELEVANT] Pif1 family helicases suppress genome instability at G-quadr |
| 3 | 12922760 [unjudged] The essential Schizosaccharomyces pombe Pfh1 DNA helicase pr | 12922760 [unjudged] The essential Schizosaccharomyces pombe Pfh1 DNA helicase pr |
| 4 | 17731780 [unjudged] The Bacteroides sp. 3_1_23 Pif1 protein is a multifunctional | 17731780 [unjudged] The Bacteroides sp. 3_1_23 Pif1 protein is a multifunctional |
| 5 | 21425864 [unjudged] A conditionally lethal yeast mutant blocked at the first ste | 21425864 [unjudged] A conditionally lethal yeast mutant blocked at the first ste |

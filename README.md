# İstanbul GraphTraffic AI

**Uncertainty-Aware Spatiotemporal Graph Learning for Multi-Horizon Traffic Forecasting**

<p align="center"><img src="docs/assets/social-preview.png" alt="İstanbul GraphTraffic AI" width="100%"></p>

<p align="center">
<a href="https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/FaramarzKowsari/istanbul-graphtraffic-ai/ci.yml?branch=main&label=CI"></a>
<a href="https://doi.org/10.17605/OSF.IO/FM5R7"><img alt="OSF" src="https://img.shields.io/badge/OSF-preregistered-2CB9A8"></a>
<a href="https://doi.org/10.5281/zenodo.21916357"><img alt="Zenodo DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21916357-blue"></a>
<a href="https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html"><img alt="Confirmatory v2" src="https://img.shields.io/badge/confirmatory%20v2-completed-brightgreen"></a>
<a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<p align="center"><strong><a href="#english">English</a> · <a href="#türkçe">Türkçe</a> · <a href="#español">Español</a></strong></p>

---

<a id="english"></a>
# English

## Research record

**İstanbul GraphTraffic AI** is a reproducible graph-based traffic-forecasting project using public İstanbul Metropolitan Municipality (İBB) hourly traffic data. Traffic locations are represented as nodes and road-travel relationships as directed edges. The repository studies temporal baselines, ST-GNNs, Dynamic Graph Transformers, adaptive adjacency, multi-horizon forecasting, uncertainty, and sensor-failure robustness.

The project now contains a **public OSF preregistration and a completed registered Confirmatory Protocol v2 analysis**.

- OSF registration DOI: [10.17605/OSF.IO/FM5R7](https://doi.org/10.17605/OSF.IO/FM5R7)
- Successful registered run: [31837216931](https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/actions/runs/31837216931)
- Permanent evidence archive: [`archive/confirmatory-v2/`](archive/confirmatory-v2/)
- Confirmatory results page: [https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html](https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html)
- Zenodo version DOI: [10.5281/zenodo.21916357](https://doi.org/10.5281/zenodo.21916357)
- Zenodo concept DOI: [10.5281/zenodo.21916358](https://doi.org/10.5281/zenodo.21916358)

## Registered primary result

**H1:** DGT Directed Road + Adaptive vs Temporal MLP at +1h.

| Metric | Result |
|---|---:|
| Mean paired daily MAE difference | **−0.0181 km/h** |
| Relative MAE difference | **−0.490%** |
| 95% hierarchical bootstrap CI | **[−0.1760, +0.1120]** |
| One-sided paired Wilcoxon p | **0.500000** |
| Registered α | **0.05** |

**Conclusion:** the preregistered H1 superiority hypothesis was **not statistically supported**.

H3 showed a raw +1h-vs-+6h horizon signal (raw p = **0.013672**), but it was **not significant after the preregistered Holm correction** (adjusted p = **0.068359**).

Confirmatory months used: **2024-05** and **2024-11**.  
`2024-02` and `2024-08` were excluded by the frozen 64-node / 98% training-coverage rule. January 2025 remains excluded from confirmatory inference because it was used in exploratory development.

## Evidence and reproducibility

Key permanent files:

- [`CONFIRMATORY_RESULTS.md`](archive/confirmatory-v2/reports/confirmatory/CONFIRMATORY_RESULTS.md)
- [`confirmatory_statistics.json`](archive/confirmatory-v2/reports/confirmatory/confirmatory_statistics.json)
- [`registered_effects.csv`](archive/confirmatory-v2/reports/confirmatory/registered_effects.csv)
- [`REGISTERED_EFFECTS_AND_PROVENANCE.md`](archive/confirmatory-v2/reports/confirmatory/REGISTERED_EFFECTS_AND_PROVENANCE.md)
- [`raw_source_provenance_manifest.json`](archive/confirmatory-v2/reports/confirmatory/raw_source_provenance_manifest.json)
- [`SHA256SUMS.txt`](archive/confirmatory-v2/SHA256SUMS.txt)

The design uses chronological splits, training-only preprocessing, fixed seeds, frozen YAML plans, automated tests, preregistered hypotheses, Holm correction for the secondary family, 10,000-replicate hierarchical bootstrap intervals, and provenance hashes.

## Models

Persistence · Historical Average · Temporal MLP · GRU · ST-GCN-style forecasting · Dynamic Graph Transformer.

Main implementation: [`src/graphtraffic/models/dynamic_graph_transformer.py`](src/graphtraffic/models/dynamic_graph_transformer.py)

## Quick start

```bash
git clone https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai.git
cd istanbul-graphtraffic-ai
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

Raw third-party İBB files are not redistributed. Synthetic data are for pipeline smoke testing only and are not real İstanbul benchmark evidence.

The project does **not** claim universal graph-model superiority or production readiness.

---

<a id="türkçe"></a>
# Türkçe

**İstanbul GraphTraffic AI**, açık İBB saatlik trafik verileriyle grafik tabanlı trafik tahminini inceleyen yeniden üretilebilir bir araştırma projesidir. Proje artık herkese açık bir **OSF ön kaydı** ve tamamlanmış **Registered Confirmatory Protocol v2** analizi içerir.

- OSF DOI: [10.17605/OSF.IO/FM5R7](https://doi.org/10.17605/OSF.IO/FM5R7)
- Başarılı kayıtlı run: [31837216931](https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/actions/runs/31837216931)
- Kalıcı arşiv: [`archive/confirmatory-v2/`](archive/confirmatory-v2/)
- Sonuç sayfası: [https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html](https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html)

### Birincil H1 sonucu

+1h DGT Directed Road + Adaptive ile Temporal MLP karşılaştırması:

- MAE farkı: **−0.0181 km/h**
- göreli MAE farkı: **−0.490%**
- %95 hiyerarşik bootstrap GA: **[−0.1760, +0.1120]**
- tek yönlü eşleştirilmiş Wilcoxon p: **0.500000**
- α = **0.05**

H1 istatistiksel olarak desteklenmedi. H3 için ham p = **0.013672**, ancak Holm düzeltmesinden sonra p = **0.068359** olduğundan ikincil sonuç anlamlı kalmadı.

Doğrulayıcı aylar **2024-05** ve **2024-11** idi. `2024-02` ve `2024-08` dondurulmuş coverage kuralıyla dışlandı; Ocak 2025 keşifsel çalışmada kullanıldığı için doğrulayıcı çıkarımdan hariçtir.

---

<a id="español"></a>
# Español

**İstanbul GraphTraffic AI** es un proyecto reproducible de predicción del tráfico basado en grafos con datos horarios públicos de İBB. El proyecto incluye ahora un **prerregistro público en OSF** y un análisis **Registered Confirmatory Protocol v2** completado.

- DOI de OSF: [10.17605/OSF.IO/FM5R7](https://doi.org/10.17605/OSF.IO/FM5R7)
- Ejecución registrada exitosa: [31837216931](https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/actions/runs/31837216931)
- Archivo permanente: [`archive/confirmatory-v2/`](archive/confirmatory-v2/)
- Página de resultados: [https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html](https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/confirmatory-results.html)

### Resultado H1 primario

Comparación a +1h: DGT Directed Road + Adaptive frente a Temporal MLP.

- diferencia de MAE: **−0.0181 km/h**
- diferencia relativa de MAE: **−0.490%**
- IC bootstrap jerárquico del 95%: **[−0.1760, +0.1120]**
- p de Wilcoxon pareado unilateral: **0.500000**
- α = **0.05**

H1 no recibió apoyo estadístico. H3 tuvo p bruto = **0.013672**, pero después de la corrección de Holm p = **0.068359**, por lo que no siguió siendo significativo.

Los meses confirmatorios utilizados fueron **2024-05** y **2024-11**. `2024-02` y `2024-08` se excluyeron por la regla congelada de cobertura; enero de 2025 permanece excluido de la inferencia confirmatoria por su uso exploratorio.

---

## Author / Yazar / Autor

**Faramarz Kowsari** — Author, Software Engineer and AI Researcher, Istanbul, Türkiye.

ORCID: https://orcid.org/0000-0003-1692-0453  
Official website: https://faramarzkowsari.github.io  
License: MIT

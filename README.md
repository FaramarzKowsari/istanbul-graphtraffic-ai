# İstanbul GraphTraffic AI

**Uncertainty-Aware Spatiotemporal Graph Learning for Multi-Horizon Traffic Forecasting**

<p align="center">
  <img src="docs/assets/social-preview.png" alt="İstanbul GraphTraffic AI" width="100%">
</p>

<p align="center">
  <a href="https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/actions">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/FaramarzKowsari/istanbul-graphtraffic-ai/ci.yml?branch=main&label=CI">
  </a>
  <a href="https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/releases/tag/v0.1.0">
    <img alt="Release" src="https://img.shields.io/badge/release-v0.1.0-blue">
  </a>
<a href="https://doi.org/10.5281/zenodo.21916357">
  <img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21916357-blue">
</a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  </a>
  <a href="CITATION.cff">
    <img alt="Citation" src="https://img.shields.io/badge/citation-CFF-blue">
  </a>
  <a href="https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/">
    <img alt="Website" src="https://img.shields.io/badge/research%20site-live-brightgreen">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Research Status" src="https://img.shields.io/badge/status-protocol%20%2B%20pipeline-orange">
</p>

<p align="center">
  <strong>
    <a href="#english">English</a>
    ·
    <a href="#türkçe">Türkçe</a>
    ·
    <a href="#español">Español</a>
  </strong>
</p>

---

<a id="english"></a>

# English

## Overview

**İstanbul GraphTraffic AI** is a reproducible research project for graph-based urban traffic forecasting in Istanbul.

The project represents traffic sensors and road relationships as a graph and investigates whether spatiotemporal graph learning can improve multi-horizon forecasting while also addressing predictive uncertainty and resilience to sensor failures.

The repository is deliberately designed as a **research pipeline rather than a results showcase**. Numerical claims about real Istanbul traffic performance are not presented until they have been produced by documented, reproducible experiments.

## Research objective

The central question is:

> Can a directed road-topology graph combined with learned adaptive connectivity improve +1h, +2h, +3h and +6h traffic forecasts while retaining calibrated uncertainty and robustness when sensors become unavailable?

## Core research questions

1. Does a directed road-topology graph outperform purely geographic sensor adjacency?
2. Does learned adaptive connectivity provide predictive value beyond a static physical graph?
3. How do ST-GNN and Graph Transformer architectures compare across +1h, +2h, +3h and +6h forecasting horizons?
4. How well calibrated are predictive intervals across congestion regimes and time periods?
5. How rapidly does forecasting performance degrade when 10%, 20% or 30% of sensors become unavailable?
6. Do learned spatial dependencies reveal the importance of Bosphorus crossings and major arterial bottlenecks?

## Research contribution

The project brings together:

- directed and weighted road-topology graphs;
- geographic kNN graph baselines;
- static and adaptive learned adjacency;
- end-to-end spatiotemporal neural forecasting;
- ST-GNN models;
- Dynamic Graph Transformer models;
- multi-horizon prediction;
- predictive uncertainty and calibration;
- random and structured sensor-failure experiments;
- ablation studies;
- confirmatory statistical analysis;
- versioned research artifacts;
- provenance and reproducibility safeguards.

## Relationship to the 2024 IBB Traffic Graph benchmark

The 2024 work *IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction Model* established an Istanbul traffic graph benchmark and used a GLEE + ExtraTrees prediction pipeline.

İstanbul GraphTraffic AI deliberately investigates a different research space through:

- directed road topology;
- weighted graph structure;
- learned dynamic dependencies;
- end-to-end spatiotemporal neural models;
- multi-horizon forecasting;
- uncertainty quantification;
- sensor-failure resilience;
- explicit ablation experiments;
- confirmatory analysis.

See:

[`docs/novelty_audit.md`](docs/novelty_audit.md)

## Data

The intended real-data workflow uses hourly Istanbul traffic observations together with sensor and location information.

Raw third-party datasets are **not redistributed** in this repository.

Downloaded source files should be placed in:

```text
data/raw/
```

The repository contains utilities for inspecting and adapting source files:

```bash
python scripts/inspect_raw_data.py data/raw/your_file.csv
python scripts/prepare_ibb_data.py --input data/raw/your_file.csv --output data/processed/traffic.csv
```

A synthetic-data generator is also provided, but only to verify that the research pipeline executes correctly from end to end:

```bash
python scripts/generate_synthetic.py --hours 336 --sensors 48
```

Synthetic outputs are not real Istanbul benchmark results and must not be interpreted as such.

## Installation

Clone the repository:

```bash
git clone https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai.git
cd istanbul-graphtraffic-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional geospatial and road-topology dependencies:

```bash
pip install -r requirements-geo.txt
```

## Fast reproducibility check

The following commands run a synthetic end-to-end smoke test:

```bash
python scripts/generate_synthetic.py --hours 240 --sensors 32
python scripts/build_graph.py --traffic data/processed/traffic.csv --mode knn
python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml
pytest -q
```

This check validates the software pipeline. It is not a scientific benchmark of Istanbul traffic.

## Research workflow

```text
Istanbul hourly traffic data
        │
        ▼
Schema audit and normalization
        │
        ├────────────► Sensor metadata
        │
        ▼
Time × sensor × feature tensor
        │
        ├────────────► Geographic kNN graph
        ├────────────► Directed OSM road-topology graph
        └────────────► Learned adaptive graph
        │
        ▼
Temporal windows
        │
        ├── Persistence
        ├── Historical Average
        ├── Temporal MLP / GRU
        ├── ST-GNN
        └── Dynamic Graph Transformer
        │
        ▼
+1h / +2h / +3h / +6h forecasts
        │
        ├── MAE / RMSE / MAPE / R²
        ├── Quantile coverage
        ├── Prediction interval width
        ├── Sensor-failure robustness
        ├── Spatial and temporal slices
        └── Ablation and statistical tests
        │
        ▼
Versioned artifacts, figures and research findings
```

## Models

The repository includes:

- Persistence baseline
- Historical Average baseline
- Temporal MLP
- GRU-based temporal forecasting
- ST-GCN-style forecasting
- Dynamic Graph Transformer

The main research model is implemented in:

[`src/graphtraffic/models/dynamic_graph_transformer.py`](src/graphtraffic/models/dynamic_graph_transformer.py)

## Reproducibility safeguards

The project includes:

- deterministic random seeds;
- chronological train / validation / test splits;
- safeguards against future-data leakage;
- YAML experiment configurations;
- automated tests;
- CI workflows;
- SHA-256 artifact manifests;
- separation of synthetic and real-data evidence;
- explicit research-status documentation;
- no manual insertion of unsupported benchmark claims.

See:

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

[`docs/research_protocol.md`](docs/research_protocol.md)

## Current research status

### v0.1.0 — Research Protocol & Executable Pipeline

The current release contains:

- research architecture;
- data adapters;
- graph builders;
- baseline models;
- neural forecasting models;
- uncertainty evaluation;
- sensor-failure experiments;
- automated tests;
- CI workflows;
- research documentation;
- GitHub Pages research website.

The project does **not yet claim**:

- final real-data benchmark superiority;
- statistically significant superiority over competing models;
- production readiness;
- validated real-world predictive-interval calibration.

---

<a id="türkçe"></a>

# Türkçe

## Genel bakış

**İstanbul GraphTraffic AI**, İstanbul trafiğinin grafik tabanlı yöntemlerle tahmin edilmesini araştıran, yeniden üretilebilir bir yapay zekâ araştırma projesidir.

Projede trafik sensörleri ve yol bağlantıları bir grafik yapısı olarak modellenir. Amaç yalnızca gelecek trafik hızını veya yoğunluğunu tahmin etmek değildir; modelin belirsizliğinin ölçülmesi ve sensör arızalarına karşı ne kadar dayanıklı olduğunun araştırılması da çalışmanın temel parçalarıdır.

Bu depo bir sonuç vitrini olarak değil, **yeniden üretilebilir bir araştırma pipeline'ı** olarak tasarlanmıştır. Gerçek İstanbul verileri üzerinde doğrulanmamış hiçbir performans değeri bilimsel sonuç olarak sunulmaz.

## Araştırma amacı

Temel araştırma sorusu şudur:

> Yönlü yol topolojisi ile öğrenilebilir adaptif bağlantıları birleştiren bir grafik modeli, belirsizlik kalibrasyonunu ve sensör arızalarına dayanıklılığı koruyarak +1, +2, +3 ve +6 saatlik trafik tahminlerini geliştirebilir mi?

## Temel araştırma soruları

1. Yönlü yol topolojisi grafiği yalnızca coğrafi yakınlığa dayalı sensör grafiğinden daha başarılı mı?
2. Öğrenilebilir adaptif bağlantılar, sabit fiziksel yol grafiğinin ötesinde ek tahmin gücü sağlıyor mu?
3. ST-GNN ve Graph Transformer modelleri +1, +2, +3 ve +6 saatlik tahminlerde nasıl karşılaştırılıyor?
4. Tahmin aralıkları farklı trafik yoğunluğu rejimlerinde ne kadar iyi kalibre ediliyor?
5. Sensörlerin %10, %20 veya %30'u kullanılamadığında tahmin performansı ne kadar bozuluyor?
6. Öğrenilen grafik yapısı Boğaz geçişlerini ve önemli ana arter darboğazlarını güçlü mekânsal bağımlılıklar olarak ortaya çıkarabiliyor mu?

## Araştırma katkısı

Proje aşağıdaki bileşenleri tek bir araştırma çerçevesinde birleştirir:

- yönlü ve ağırlıklı yol topolojisi grafikleri;
- coğrafi kNN baseline grafikleri;
- sabit ve öğrenilebilir adaptif bağlantılar;
- uçtan uca zamansal-mekânsal sinir ağı tahmini;
- ST-GNN modelleri;
- Dynamic Graph Transformer;
- çok ufuklu trafik tahmini;
- belirsizlik ölçümü ve kalibrasyon;
- rastgele ve yapısal sensör arızası deneyleri;
- ablation analizleri;
- doğrulayıcı istatistiksel analiz;
- sürümlenmiş araştırma çıktıları;
- provenance ve yeniden üretilebilirlik kontrolleri.

## 2024 IBB Traffic Graph benchmark çalışmasından farkı

2024 tarihli *IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction Model* çalışması İstanbul trafik sensörleri için bir grafik benchmark'ı oluşturmuş ve GLEE + ExtraTrees yaklaşımını kullanmıştır.

İstanbul GraphTraffic AI ise farklı bir araştırma alanına odaklanır:

- yönlü yol topolojisi;
- ağırlıklı grafik yapısı;
- öğrenilebilir dinamik bağlantılar;
- uçtan uca spatiotemporal neural forecasting;
- çok ufuklu tahmin;
- uncertainty quantification;
- sensör arızası dayanıklılığı;
- açık ablation deneyleri;
- doğrulayıcı analiz.

Ayrıntılar:

[`docs/novelty_audit.md`](docs/novelty_audit.md)

## Veri

Gerçek araştırma akışı, İstanbul'un saatlik trafik gözlemleri ile sensör ve konum bilgilerinin kullanılmasını hedefler.

Harici ham veri dosyaları bu depoda yeniden dağıtılmaz.

İndirilen kaynak dosyaları şu klasöre yerleştirilmelidir:

```text
data/raw/
```

Dosyaları incelemek ve proje formatına dönüştürmek için:

```bash
python scripts/inspect_raw_data.py data/raw/your_file.csv
python scripts/prepare_ibb_data.py --input data/raw/your_file.csv --output data/processed/traffic.csv
```

Pipeline'ın uçtan uca doğru çalıştığını kontrol etmek için sentetik veri üreticisi de bulunmaktadır:

```bash
python scripts/generate_synthetic.py --hours 336 --sensors 48
```

Sentetik verilerden elde edilen sonuçlar gerçek İstanbul trafik benchmark sonuçları değildir.

## Kurulum

Depoyu klonlayın:

```bash
git clone https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai.git
cd istanbul-graphtraffic-ai
```

Sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Ana bağımlılıkları kurun:

```bash
pip install -r requirements.txt
pip install -e .
```

İsteğe bağlı coğrafi ve yol topolojisi bağımlılıkları:

```bash
pip install -r requirements-geo.txt
```

## Hızlı yeniden üretilebilirlik testi

Aşağıdaki komutlar sentetik veri üzerinde uçtan uca bir smoke test çalıştırır:

```bash
python scripts/generate_synthetic.py --hours 240 --sensors 32
python scripts/build_graph.py --traffic data/processed/traffic.csv --mode knn
python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml
pytest -q
```

Bu test yazılım pipeline'ını doğrular; gerçek İstanbul trafik performansını ölçmez.

## Araştırma akışı

```text
İstanbul saatlik trafik verileri
        │
        ▼
Şema denetimi ve normalizasyon
        │
        ├────────────► Sensör metaverisi
        │
        ▼
Zaman × sensör × özellik tensörü
        │
        ├────────────► Coğrafi kNN grafiği
        ├────────────► Yönlü OSM yol topolojisi grafiği
        └────────────► Öğrenilebilir adaptif grafik
        │
        ▼
Zamansal pencereler
        │
        ├── Persistence
        ├── Historical Average
        ├── Temporal MLP / GRU
        ├── ST-GNN
        └── Dynamic Graph Transformer
        │
        ▼
+1 / +2 / +3 / +6 saat tahminleri
        │
        ├── MAE / RMSE / MAPE / R²
        ├── Quantile coverage
        ├── Tahmin aralığı genişliği
        ├── Sensör arızası dayanıklılığı
        ├── Mekânsal ve zamansal analizler
        └── Ablation ve istatistiksel testler
        │
        ▼
Sürümlenmiş çıktılar, grafikler ve araştırma bulguları
```

## Modeller

Depoda bulunan temel model aileleri:

- Persistence baseline
- Historical Average baseline
- Temporal MLP
- GRU tabanlı zamansal tahmin
- ST-GCN tarzı grafik tahmin modeli
- Dynamic Graph Transformer

Ana araştırma modeli:

[`src/graphtraffic/models/dynamic_graph_transformer.py`](src/graphtraffic/models/dynamic_graph_transformer.py)

## Yeniden üretilebilirlik önlemleri

Projede:

- deterministik random seed değerleri;
- kronolojik train / validation / test ayrımı;
- gelecek verisinin eğitime sızmasını engelleyen kontroller;
- YAML deney konfigürasyonları;
- otomatik testler;
- CI workflow'ları;
- SHA-256 artifact manifestleri;
- sentetik ve gerçek sonuçların açık biçimde ayrılması;
- desteklenmeyen benchmark değerlerinin manuel olarak eklenmemesi

uygulanmaktadır.

Ayrıntılar:

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

[`docs/research_protocol.md`](docs/research_protocol.md)

## Güncel araştırma durumu

### v0.1.0 — Araştırma Protokolü ve Çalıştırılabilir Pipeline

Mevcut sürüm şunları içerir:

- araştırma mimarisi;
- veri adaptörleri;
- grafik oluşturucular;
- baseline modeller;
- sinir ağı tahmin modelleri;
- belirsizlik değerlendirmesi;
- sensör arızası deneyleri;
- otomatik testler;
- CI workflow'ları;
- araştırma belgeleri;
- GitHub Pages araştırma sitesi.

Bu aşamada:

- gerçek veride nihai benchmark üstünlüğü;
- istatistiksel olarak kanıtlanmış model üstünlüğü;
- production readiness;
- gerçek dünya tahmin aralıklarının kesin kalibrasyonu

iddia edilmemektedir.

---

<a id="español"></a>

# Español

## Descripción general

**İstanbul GraphTraffic AI** es un proyecto de investigación reproducible dedicado a la predicción del tráfico urbano de Estambul mediante modelos basados en grafos.

El proyecto representa los sensores de tráfico y las relaciones de la red viaria como un grafo y estudia si el aprendizaje espaciotemporal puede mejorar las predicciones a múltiples horizontes.

El objetivo no consiste únicamente en producir una predicción de velocidad o congestión. También se analiza la incertidumbre predictiva y la resistencia del modelo cuando parte de los sensores deja de estar disponible.

El repositorio se ha diseñado como una **infraestructura de investigación reproducible**, no como una exposición de resultados sin verificar.

## Objetivo de investigación

La pregunta central es:

> ¿Puede un grafo dirigido basado en la topología viaria, combinado con conexiones adaptativas aprendidas, mejorar las predicciones de tráfico a +1, +2, +3 y +6 horas manteniendo una incertidumbre bien calibrada y robustez frente a fallos de sensores?

## Preguntas principales

1. ¿Un grafo dirigido basado en la red viaria supera a un grafo construido únicamente mediante proximidad geográfica?
2. ¿Las conexiones adaptativas aprendidas aportan capacidad predictiva adicional al grafo físico estático?
3. ¿Cómo se comparan los modelos ST-GNN y Graph Transformer en horizontes de +1, +2, +3 y +6 horas?
4. ¿Hasta qué punto están bien calibrados los intervalos predictivos en distintos regímenes de congestión?
5. ¿Cómo se degrada el rendimiento cuando falla el 10 %, el 20 % o el 30 % de los sensores?
6. ¿Las dependencias espaciales aprendidas permiten identificar la importancia de los cruces del Bósforo y de las principales arterias de tráfico?

## Contribución de investigación

El proyecto integra:

- grafos dirigidos y ponderados de la red viaria;
- grafos geográficos kNN como baseline;
- adyacencia estática y adaptativa aprendida;
- predicción neuronal espaciotemporal de extremo a extremo;
- modelos ST-GNN;
- Dynamic Graph Transformer;
- predicción multihorizonte;
- cuantificación y calibración de incertidumbre;
- experimentos de fallos de sensores aleatorios y estructurados;
- estudios de ablación;
- análisis estadístico confirmatorio;
- artefactos de investigación versionados;
- controles de procedencia y reproducibilidad.

## Diferencias respecto al benchmark IBB Traffic Graph de 2024

El trabajo de 2024 *IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction Model* estableció un benchmark de tráfico basado en grafos para Estambul y utilizó un pipeline GLEE + ExtraTrees.

İstanbul GraphTraffic AI investiga un espacio diferente mediante:

- topología viaria dirigida;
- estructura de grafo ponderada;
- dependencias dinámicas aprendidas;
- modelos neuronales espaciotemporales de extremo a extremo;
- predicción a múltiples horizontes;
- cuantificación de incertidumbre;
- resiliencia ante fallos de sensores;
- experimentos explícitos de ablación;
- análisis confirmatorio.

Más información:

[`docs/novelty_audit.md`](docs/novelty_audit.md)

## Datos

El flujo previsto para los experimentos reales utiliza observaciones horarias del tráfico de Estambul junto con información de sensores y localización.

Los datos externos originales **no se redistribuyen** en este repositorio.

Los archivos descargados deben colocarse en:

```text
data/raw/
```

Para inspeccionar y adaptar los archivos:

```bash
python scripts/inspect_raw_data.py data/raw/your_file.csv
python scripts/prepare_ibb_data.py --input data/raw/your_file.csv --output data/processed/traffic.csv
```

También se incluye un generador de datos sintéticos únicamente para comprobar que el pipeline completo funciona:

```bash
python scripts/generate_synthetic.py --hours 336 --sensors 48
```

Los resultados obtenidos con datos sintéticos no representan resultados reales del tráfico de Estambul.

## Instalación

Clone el repositorio:

```bash
git clone https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai.git
cd istanbul-graphtraffic-ai
```

Cree un entorno virtual:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Instale las dependencias principales:

```bash
pip install -r requirements.txt
pip install -e .
```

Dependencias geoespaciales y de topología viaria opcionales:

```bash
pip install -r requirements-geo.txt
```

## Comprobación rápida de reproducibilidad

Los siguientes comandos ejecutan una prueba sintética de extremo a extremo:

```bash
python scripts/generate_synthetic.py --hours 240 --sensors 32
python scripts/build_graph.py --traffic data/processed/traffic.csv --mode knn
python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml
pytest -q
```

Esta prueba valida el pipeline de software. No constituye un benchmark científico del tráfico real de Estambul.

## Flujo de investigación

```text
Datos horarios de tráfico de Estambul
        │
        ▼
Auditoría del esquema y normalización
        │
        ├────────────► Metadatos de sensores
        │
        ▼
Tensor tiempo × sensor × características
        │
        ├────────────► Grafo geográfico kNN
        ├────────────► Grafo viario dirigido OSM
        └────────────► Grafo adaptativo aprendido
        │
        ▼
Ventanas temporales
        │
        ├── Persistence
        ├── Historical Average
        ├── Temporal MLP / GRU
        ├── ST-GNN
        └── Dynamic Graph Transformer
        │
        ▼
Predicciones +1 h / +2 h / +3 h / +6 h
        │
        ├── MAE / RMSE / MAPE / R²
        ├── Cobertura de cuantiles
        ├── Anchura de intervalos predictivos
        ├── Robustez frente a fallos de sensores
        ├── Análisis espacial y temporal
        └── Ablación y pruebas estadísticas
        │
        ▼
Artefactos versionados, figuras y resultados
```

## Modelos

El repositorio incluye:

- Persistence baseline
- Historical Average baseline
- Temporal MLP
- modelo temporal basado en GRU
- modelo de estilo ST-GCN
- Dynamic Graph Transformer

El modelo principal se encuentra en:

[`src/graphtraffic/models/dynamic_graph_transformer.py`](src/graphtraffic/models/dynamic_graph_transformer.py)

## Garantías de reproducibilidad

El proyecto incorpora:

- semillas aleatorias deterministas;
- separación cronológica train / validation / test;
- protección frente a fugas de información futura;
- configuraciones experimentales YAML;
- pruebas automatizadas;
- workflows de CI;
- manifiestos SHA-256;
- separación clara entre resultados sintéticos y reales;
- prohibición de introducir manualmente resultados de benchmark no respaldados por experimentos.

Consulte:

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

[`docs/research_protocol.md`](docs/research_protocol.md)

## Estado actual de la investigación

### v0.1.0 — Protocolo de investigación y pipeline ejecutable

La versión actual contiene:

- arquitectura de investigación;
- adaptadores de datos;
- constructores de grafos;
- modelos baseline;
- modelos neuronales de predicción;
- evaluación de incertidumbre;
- experimentos de fallos de sensores;
- pruebas automatizadas;
- CI;
- documentación científica;
- sitio de investigación en GitHub Pages.

En esta fase no se afirma:

- superioridad definitiva en benchmarks de datos reales;
- superioridad estadística confirmada;
- preparación para producción;
- calibración definitiva de intervalos predictivos reales.

---

# Repository Structure · Depo Yapısı · Estructura del Repositorio

```text
configs/                 Experiment configurations
data/                     Data instructions and processed-data workspace
docs/                     GitHub Pages and research documentation
notebooks/                Research notebooks
reports/                  Generated figures and tables
scripts/                  Reproducible command-line utilities
src/graphtraffic/         Core research source code
tests/                    Automated tests
.github/workflows/        CI and research workflows
```

---

# Research Archive · Araştırma Arşivi · Archivo de Investigación

**Release**

[`v0.1.0 — Research Protocol & Executable Pipeline`](https://github.com/FaramarzKowsari/istanbul-graphtraffic-ai/releases/tag/v0.1.0)

**Version DOI — exact archived v0.1.0 release**

[10.5281/zenodo.21916357](https://doi.org/10.5281/zenodo.21916357)

**Concept DOI — all project versions**

[10.5281/zenodo.21916358](https://doi.org/10.5281/zenodo.21916358)

**Research Website**

https://faramarzkowsari.github.io/istanbul-graphtraffic-ai/

---

# Citation · Atıf · Citación

For reproducible research, cite the exact archived release.

Yeniden üretilebilir araştırma için belirli arşivlenmiş sürümü kullanın.

Para garantizar la reproducibilidad, cite la versión archivada concreta.

> Kowsari, F. (2026). *İstanbul GraphTraffic AI* (Version 0.1.0). Zenodo. https://doi.org/10.5281/zenodo.21916357

Machine-readable citation metadata:

[`CITATION.cff`](CITATION.cff)

---

# Author · Yazar · Autor

<table>
<tr>
<td width="135">
<img src="https://avatars.githubusercontent.com/u/105053743?v=4&s=512"
     width="115"
     alt="Faramarz Kowsari">
</td>
<td>

<strong>Faramarz Kowsari</strong><br>
Author · Software Engineer · AI Researcher<br>
Istanbul, Türkiye<br><br>

<a href="https://faramarzkowsari.github.io">Official Website</a> ·
<a href="https://github.com/FaramarzKowsari">GitHub</a> ·
<a href="https://orcid.org/0000-0003-1692-0453">ORCID</a> ·
<a href="https://scholar.google.com/citations?user=G7tP5WMAAAAJ&hl=en">Google Scholar</a>

</td>
</tr>
</table>

---

# License · Lisans · Licencia

The source code in this repository is released under the **MIT License**.

Bu depodaki kaynak kod **MIT Lisansı** altında yayımlanmaktadır.

El código fuente de este repositorio se distribuye bajo la **licencia MIT**.

External datasets retain their own licenses and terms. OpenStreetMap-derived data remain subject to the applicable OpenStreetMap attribution and ODbL requirements.

See:

[`LICENSE`](LICENSE)

[`docs/data_sources.md`](docs/data_sources.md)

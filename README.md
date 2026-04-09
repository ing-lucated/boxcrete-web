[README.md](https://github.com/user-attachments/files/26600497/README.md)
# BOxCrete Web App 🏗️

App Flask per la previsione probabilistica della resistenza del calcestruzzo e del Global Warming Potential (GWP), basata sul framework [BOxCrete](https://github.com/facebookresearch/SustainableConcrete) di Meta Research.

## Avvio rapido

```bash
# 1. Installa le dipendenze base (senza PyTorch)
pip install -r requirements.txt

# 2. Avvia il server
python app.py

# 3. Apri il browser su
http://localhost:5000
```

## Modalità modello

| Modalità | Requisiti | Accuratezza |
|---|---|---|
| **Surrogate GP** | solo numpy/scipy | Empirica (Abrams law + fisica) |
| **BOxCrete GP** | torch + gpytorch + botorch + boxcrete | Alta (dati sperimentali Meta) |

### Abilitare il modello BOxCrete reale

```bash
pip install torch gpytorch botorch
pip install git+https://github.com/facebookresearch/SustainableConcrete.git
python app.py   # detecta automaticamente il pacchetto
```

## Funzionalità

- **Curva di resistenza** 1–90 giorni con intervallo di confidenza 95%
- **Milestone** a 1, 7, 28, 90 giorni
- **GWP** (kg CO₂-eq/m³) calcolato con fattori di emissione da letteratura
- **Visualizzazione composizione legante** (cemento, fly ash, GGBS, silica fume)
- **Insight automatico** con classe di resistenza indicativa e suggerimenti di sostenibilità

## API

```
POST /api/predict
Content-Type: application/json

{
  "cement": 350,
  "fly_ash": 100,
  "slag": 0,
  "silica_fume": 0,
  "water": 175,
  "aggregate_fine": 750,
  "aggregate_coarse": 1000,
  "admixture": 2
}
```

## Riferimenti

- Paper: [Sustainable Concrete via Bayesian Optimization](https://arxiv.org/abs/2310.18288)
- Repo originale: [facebookresearch/SustainableConcrete](https://github.com/facebookresearch/SustainableConcrete)

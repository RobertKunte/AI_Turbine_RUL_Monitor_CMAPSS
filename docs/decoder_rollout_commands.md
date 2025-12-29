# Transformer Decoder Rollout - Commands

## Step 3: Run Experiments (FD001-FD003)

```bash
python -u run_experiments.py --experiments \
  fd001_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar,\
  fd002_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar,\
  fd003_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar \
  --device cuda
```

**Note:** Alle Runs haben `max_rul = 125` (cap 125) gesetzt.

## Step 4: Compare Results per Dataset

Nach Abschluss der Runs:

```bash
# FD001 Vergleich
python -m src.tools.compare_decoder_experiments --dataset FD001

# FD002 Vergleich
python -m src.tools.compare_decoder_experiments --dataset FD002

# FD003 Vergleich
python -m src.tools.compare_decoder_experiments --dataset FD003

# FD004 Vergleich (bereits vorhanden)
python -m src.tools.compare_decoder_experiments --dataset FD004
```

## Step 5: Seed-Check (FD004 Repro)

**Ziel:** Reproduzierbarkeit mit Seed 43 prüfen.

### 5.1: Config erstellen

Die Config `fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43` muss in `src/experiment_configs.py` erstellt werden:

```python
def get_fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43_config() -> ExperimentConfig:
    """
    Transformer AR decoder variant with seed 43 for reproducibility check.
    
    Identical to fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar except random_seed=43.
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_config())
    cfg["experiment_name"] = "fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43"
    
    # Change ONLY random_seed
    cfg.setdefault("training_params", {})
    cfg["training_params"]["random_seed"] = 43
    
    return cfg
```

Und in `get_experiment_by_name()` registrieren:

```python
if experiment_name == "fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43":
    return get_fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43_config()
```

### 5.2: Run ausführen

```bash
python -u run_experiments.py --experiments fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43 --device cuda
```

### 5.3: Vergleiche

```bash
# CONTROL seed42 vs TF_AR seed42 (bereits vorhanden)
python -m src.tools.compare_decoder_experiments --dataset FD004 \
  --control fd004_wm_v1_p0_softcap_k3_hm_pad \
  --treatment fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar

# CONTROL seed42 vs TF_AR seed43 (Stabilität grob)
python -m src.tools.compare_decoder_experiments --dataset FD004 \
  --control fd004_wm_v1_p0_softcap_k3_hm_pad \
  --treatment fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43

# TF_AR seed42 vs TF_AR seed43 (Reproduzierbarkeit)
python -m src.tools.compare_decoder_experiments --dataset FD004 \
  --control fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar \
  --treatment fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_seed43
```

## Step 6: Cross-Attention Variante (Stretch)

**Voraussetzung:** Encoder muss Sequenz ausgeben können (nicht nur Token).

### 6.1: Prüfen ob Encoder-Seq verfügbar

Prüfe in `src/models/world_model.py` ob `UniversalEncoderV2` eine Sequenz ausgeben kann:

```python
# In WorldModelUniversalV3.forward():
# Prüfe ob enc_seq verfügbar ist (nicht nur enc_token)
```

Falls aktuell nur `(B, d_model)` Token verfügbar ist:
- **Skip Step 6** - Cross-Attention bringt wenig ohne Encoder-Sequenz

Falls Sequenz verfügbar ist:

### 6.2: Config erstellen

Für FD004 testen:

```python
def get_fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn_config() -> ExperimentConfig:
    """
    Transformer AR decoder variant with cross-attention.
    
    Identical to fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar except decoder_type="tf_ar_xattn".
    """
    cfg = copy.deepcopy(get_fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_config())
    cfg["experiment_name"] = "fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn"
    
    # Change decoder_type to cross-attention variant
    wmp = cfg.setdefault("world_model_params", {})
    wmp["decoder_type"] = "tf_ar_xattn"
    wmp["max_rul"] = 125
    
    return cfg
```

### 6.3: Run ausführen

```bash
python -u run_experiments.py --experiments fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn --device cuda
```

### 6.4: Vergleich

```bash
# TF_AR (self-attn) vs TF_AR_XATTN (cross-attn)
python -m src.tools.compare_decoder_experiments --dataset FD004 \
  --control fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar \
  --treatment fd004_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn
```

### 6.5: Rollout auf FD001-FD003 (falls erfolgreich)

Falls FD004 Cross-Attention erfolgreich ist, analog Configs für FD001-FD003 erstellen:

```bash
python -u run_experiments.py --experiments \
  fd001_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn,\
  fd002_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn,\
  fd003_wm_v1_p0_softcap_k3_hm_pad_dec_tf_ar_xattn \
  --device cuda
```

## Zusammenfassung

- **Step 3:** FD001-FD003 Runs (mit `max_rul=125`)
- **Step 4:** Vergleiche pro Dataset
- **Step 5:** Seed-Check (FD004 seed43) - **Optional**
- **Step 6:** Cross-Attention Variante - **Stretch** (nur wenn Encoder-Seq verfügbar)


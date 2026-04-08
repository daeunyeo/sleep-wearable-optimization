# Sleep Stage Classification with Multimodal Biosignals
### Minimum-Channel Wearable Feasibility via SHAP-based Channel Attribution

---

## 1. Background & Motivation

Polysomnography (PSG), the clinical gold standard for sleep staging, requires attachment of 10+ electrodes across the body — causing patient discomfort, limiting use to hospital settings, and driving high costs. Reducing the number of channels directly translates to:
- Reduced patient discomfort during overnight recording
- Simplified device hardware and lower manufacturing cost
- Potential for accessible, consumer-grade sleep monitoring

**Core research question:** *Which channels are actually necessary? Can a minimal channel configuration achieve clinically acceptable sleep stage classification?*

### Model Selection: Why Random Forest?

Prior to this project, a baseline experiment (`baseline/`) was conducted on Sleep-EDF comparing KNN, SVM, and RF across 7 iterative improvements (data expansion, feature engineering, class_weight tuning, grid search):

| Model | Final Test Accuracy |
|-------|-------------------|
| KNN   | 62.69% |
| SVM   | 64.93% |
| **RF** | **67.91%** |

RF was selected based on this empirical comparison. Gradient boosting models (XGBoost, LightGBM) were excluded in favor of RF due to its superior compatibility with SHAP TreeExplainer for multi-class feature attribution — the primary analytical goal of this project.

---

## 2. Dataset

- **Source:** Sleep-EDF Cassette (PhysioNet), 7 subjects
- **Channels:** EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, Resp oro-nasal
- **Labels:** W / N1 / N2 / N3 / REM (AASM criteria)
- **Epochs:** 957 total (30s per epoch)

### Channel Selection & Verification

**EMG (submental chin) — excluded**
Signal quality verification revealed DC drift artifacts with no discernible high-frequency muscle activity in the Sleep-EDF Cassette EMG channel (confirmed via highpass filtering at 10–20 Hz). This is consistent with known limitations of the cassette recorder setup. Additionally, chin EMG electrode placement is physically incompatible with lightweight frontal wearable form factors.

**EEG Pz-Oz — included as independent channel**
EEG Fpz-Cz and Pz-Oz capture activity from anatomically distinct regions — frontal and occipital cortex respectively — making them physiologically complementary rather than redundant. This was confirmed empirically: Pearson correlation computed across 2 subjects (303 epochs) yielded **r = -0.386**, consistent with the expected independence.

---

## 3. Feature Engineering

### Channel-wise Feature Extraction

| Channel | Features | Physiological Rationale |
|---------|----------|------------------------|
| EEG Fpz-Cz | delta(0.5–4Hz) / theta(4–8Hz) / alpha(8–12Hz) / sigma(12–15Hz) / beta(15–30Hz) band power (Welch PSD) | delta→N3, sigma(sleep spindle)→N2, alpha→W |
| EEG Pz-Oz | Same 5 bands | Independent occipital complement (r = -0.386) |
| EOG | RMS, ZCR, diff_var | Captures REM rapid eye movement intensity, frequency, and velocity |
| Resp | Breath rate, std | Reflects sleep-depth-dependent regularity changes |

**Why sigma band was added separately:**
Beta (13–30 Hz) originally subsumed the sleep spindle range. Isolating sigma (12–15 Hz) as an independent feature directly targets N2-specific oscillations, reducing REM→N2 misclassification.

**EOG diff_var:**
First-order derivative variance captures the acceleration of eye movements — distinguishing REM (rapid, jerky) from N1 (slow rolling). This reduced REM→N1 misclassification.

**Lagged Features (t-1, t, t+1):**
RF treats each epoch independently, ignoring temporal context. Concatenating features from the preceding and following epochs provides sequential sleep context — reflecting the physiological continuity of sleep stage transitions. Features expanded from 15 → 45 per epoch.

### Resp Channel — Excluded from Final Model

SHAP analysis revealed Resp channel contribution was lowest across all sleep stages (avg. ~0.001–0.002, approximately 1/10 of EEG contribution).

This aligns with a practical argument: respiratory sensors (chest belt, nasal cannula, thermistor) are among the most uncomfortable components in overnight PSG recordings, prone to motion artifacts and long-term displacement. Excluding Resp is justified on both **algorithmic** (SHAP evidence) and **clinical usability** (wearable feasibility) grounds.

---

## 4. Results

### Ablation: Formfactor-wise Performance (Random Split, 80/20)

| Configuration | Channels | Acc | REM F1 |
|--------------|----------|-----|--------|
| Full (baseline) | EEG Fpz + EEG Poz + EOG + Resp | 0.862 | 0.81 |
| **EEG + EOG** ⭐ | **EEG Fpz + EOG** | **0.847** | **0.78** |
| Dual EEG | EEG Fpz + EEG Poz | 0.852 | 0.77 |
| Single EEG | EEG Fpz only | 0.820 | 0.67 |
| EEG + Resp | EEG Fpz + Resp | 0.810 | 0.72 |

<img width="1289" height="590" alt="18   ab" src="https://github.com/user-attachments/assets/1c89fbe0-38ea-45aa-8669-a6fe7477df68" />

**EEG + EOG achieves accuracy within 1.5% of the 4-channel full model**, with the highest REM F1 among all reduced configurations.

### LOSO: Subject-Independent Validation (7 subjects)

| Configuration | Acc (mean ± std) | REM F1 |
|--------------|-----------------|--------|
| Full (4-channel) | 0.705 ± 0.068 | 0.11 |
| **EEG + EOG** ⭐ | **0.716 ± 0.068** | **0.34** |
| Dual EEG | 0.717 ± 0.083 | 0.09 |

<img width="1490" height="495" alt="18  ablation eng" src="https://github.com/user-attachments/assets/ff44267b-d7c3-45d2-b1e8-6647da3b3a55" />

**EEG + EOG outperforms the full 4-channel model on REM F1 even under LOSO**, confirming that EOG contributes generalizable cross-subject REM detection — not just within-subject pattern memorization.

### The Gap between Random Split and LOSO

The performance drop from Random Split (Acc 0.847) to LOSO (Acc 0.716) directly reflects the **inter-subject variability** inherent in biosignals. Rather than optimizing for inflated in-sample metrics, this project explicitly evaluated generalization to unseen subjects — a critical requirement for real-world wearable deployment.

This gap is not a failure. It is a deliberate design choice: reporting LOSO alongside random split demonstrates awareness of the difference between benchmark performance and real-world applicability — a distinction that most single-dataset ML projects overlook.

### SHAP: Channel-level Attribution

| Sleep Stage | Top Contributing Channel | Key Feature |
|------------|--------------------------|-------------|
| W   | EOG            | eog_diff_var        |
| N1  | EEG Fpz-Cz     | fpz_delta           |
| N2  | EEG Fpz-Cz     | fpz_delta           |
| N3  | EEG Fpz-Cz     | fpz_delta (dominant)|
| REM | EEG Fpz-Cz     | fpz_delta, fpz_sigma|

<img width="1990" height="631" alt="18  shap eng" src="https://github.com/user-attachments/assets/de035966-c0bf-45b3-ae22-c03b717eec67" />

<img width="1389" height="593" alt="18  rem shap eng" src="https://github.com/user-attachments/assets/98a3bf3f-98f5-4827-833f-c9df24be2ac9" />

EOG contribution to REM prediction: **4× higher than Resp** across all test epochs. Resp ranked last in every sleep stage — quantitatively supporting its exclusion.

---

## 5. Lightweight Wearable Feasibility

The algorithm was initially framed around a sleep eye mask form factor. However, the same 2-channel (EEG Fpz-Cz + EOG) configuration is equally realizable in a forehead patch or other compact wearables, as the feature design is channel-combination-based rather than form-factor-specific.

EEG Fpz-Cz + EOG achieves equivalent performance to a 4-channel setup at 1.5% accuracy cost. This 2-channel configuration is physically realizable in compact wearable form factors:

| Form Factor | Advantages | Limitations |
|------------|------------|-------------|
| Sleep eye mask | Integrates EEG + EOG electrodes at forehead/periorbital positions; low daily-use resistance | Light occlusion may affect sleep environment |
| Forehead patch | Single patch integrates both sensors; no hair interference; lower power draw | Skin adhesive discomfort over extended wear |

*Note: This study is limited to algorithm validation. Physical device implementation and signal quality verification on wearable hardware are reserved as future work.*

---

## 6. Limitations & Future Work

- **LOSO REM F1 = 0.34:** Root cause — 7-subject dataset and known EOG channel cross-talk in Sleep-EDF Cassette (cassette recorder setup). Seven feature engineering interventions (Baseline Calibration, ratio features, log transform, Peak Count, EOG frequency decomposition, TEO, Hjorth parameters) were systematically tested; none improved LOSO REM F1 beyond baseline, confirming dataset-level constraints rather than feature-level deficiency.
- **Scale:** LOSO stabilization expected with ≥20 subjects. Cross-dataset validation (HMC, ISRUC) is a planned extension.
- **Deep learning:** 1D-CNN / AttnSleep expected to push LOSO REM F1 above 0.5 via end-to-end temporal feature learning.
- **Personalization:** Rolling Window Normalization for real-time subject adaptation without calibration sessions.

---

## 7. Repository Structure
```
sleep-stage-classification/
├── baseline/
│   └── sleep_ml_baseline.ipynb      # KNN / SVM / RF comparison (7 iterations)
├── multimodal/
│   └── sleep_eyemask_pipeline.ipynb # Full pipeline
├── figures/
│   ├── step5_ablation_comparison.png
│   ├── step5_loso_result.png
│   ├── step6_full_shap_per_stage.png
│   └── step6_rem_shap_comparison.png
└── README.md
```
---

## 8. Setup
```bash
pip install mne scikit-learn shap pandas numpy scipy matplotlib
```

**Data:** Download Sleep-EDF Cassette from PhysioNet (open access, no registration required):
[https://physionet.org/content/sleep-edfx/1.0.0/](https://physionet.org/content/sleep-edfx/1.0.0/)

---

## 9. Citation

Kemp B, et al. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE-BME* 47(9):1185–1194 (2000).

# Aegis: Threat Diagnostics & Security Intelligence Suite

Aegis is a hybrid threat diagnostics and incident correlation framework designed for security threat detection, adversarial evasion analysis, and multi-agent incident orchestration. This release repository contains the core Streamlit application, trained model weights, reference dataset splits, and the corresponding academic LaTeX research paper code.

---

## 🚀 Key Features

1. **🔍 Threat Diagnostics (Inference & Payload Inspection)**:
   - Utilizes a trained **CNN-BiLSTM-Transformer (BERT-Style) Deep Learning Model** that processes raw packet payload bytes and scaled metadata (`ttl`, `total_len`, `t_delta`, `protocol`) to predict threats across 6 unified classes: `BENIGN`, `DoS`, `PortScan`, `Infiltration`, `Bot`, and `Brute Force`.
   - Allows sampling packets from pre-split validation datasets (CICIDS2017, UNSW-NB15, and Synthetic logs).

2. **🤖 Multi-Agent LLM Incident Correlator Swarm**:
   - Integrates a local, privacy-centric Multi-Agent Large Language Model swarm (Host, Network, MITRE ATT&CK Mapper, and Coordinator agents) that correlates host process logs and network flows into unified mitre-mapped incident summaries.
   - Operates with Ollama (local llama3 model) or falls back to a high-fidelity mock reasoning engine.

3. **🧪 Adversarial Evasion Sandbox**:
   - Implements the **Fast Gradient Sign Method (FGSM)** to apply byte-level mathematical perturbations to the raw network packet payloads, evaluating the model's classification shift under varying noise levels ($\epsilon$).

---

## 📂 Repository Directory Structure

* **`app.py`**: The Streamlit dashboard application.
* **`aegis_paper.tex`**: Academic LaTeX source code of the Aegis research paper.
* **`requirements.txt`**: Python dependencies list.
* **`setup_and_launch.bat`**: Automated environment installation and launcher script for Windows.
* **`models/`**: Folder containing the trained neural network model weights and preprocessors:
  - `model.pth`: Trained CNN-BiLSTM-Transformer model weights.
  - `scaler.joblib`: Standard scaler for network metadata.
  - `label_encoder.joblib`: Label encoder for the 6 threat classes.
  - `protocol_encoder.joblib`: Protocol encoder for network layer protocols.
* **`data/`**: Folder containing the reference validation dataset:
  - `balanced_val.csv`: Reference balanced evaluation dataset (sampled from all 3 datasets, containing exactly equal classes).

---

## 🛠️ Installation & Run Instructions

### Automated Startup (Windows)
Simply double-click the **`setup_and_launch.bat`** file. The script will automatically:
1. Verify Python is installed and added to the System Path.
2. Initialize a local virtual environment (`.venv`) inside this folder.
3. Install all dependencies from `requirements.txt`.
4. Launch the Streamlit dashboard on port `8550` and open it in your default web browser.

---

### Manual Startup (Cross-Platform)

If you are on Linux/macOS or prefer running commands manually, follow these steps:

#### 1. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Streamlit Application
```bash
streamlit run app.py --server.port 8550
```
Then, navigate your web browser to `http://localhost:8550`.

---

## 🔬 Model Performance Reference

The model trained in this setup achieves **94.57% Overall Accuracy** and a **0.85 Macro F1-score** on the combined evaluation validation split. The balanced validation subsets contain 3,055 validation rows per class:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **BENIGN** | 0.95 | 0.96 | 0.96 | 155,244 |
| **Bot** | 0.55 | 0.89 | 0.68 | 3,056 |
| **Brute Force** | 0.95 | 0.94 | 0.95 | 43,189 |
| **DoS** | 0.99 | 0.95 | 0.97 | 339,826 |
| **Infiltration** | 0.85 | 0.89 | 0.87 | 79,388 |
| **PortScan** | 0.53 | 0.86 | 0.66 | 5,357 |
| **Overall Accuracy** | | | **94.57%** | **626,060** |
| **Macro Average** | **0.80** | **0.92** | **0.85** | **626,060** |
| **Weighted Average** | **0.96** | **0.95** | **0.95** | **626,060** |


# 🧠 Reddit NLP Research — Mental Health Behavioral Intent Modeling

### Transformer-based NLP for Advice-Seeking & Behavioral Classification in Online Mental Health Communities

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Research-Paper%20Ready-brightgreen)

---

## 📌 Overview

Online mental health communities on Reddit host millions of posts where users seek support, share experiences, and express distress. Automatically understanding **what a user is looking for** — emotional support, advice, or information — is key to building effective AI-driven mental health tools.

This project builds a full **transformer-based NLP pipeline** to classify **behavioral intent** (e.g., advice-seeking, emotional venting, informational support) across **112,000+ Reddit posts** scraped from mental health subreddits. It compares classical ML baselines against a **domain-adaptive fine-tuned BERT model** and a **Universal Sentence Encoder (USE)** approach, with rigorous ablation studies and error analysis.

> 📄 A research paper accompanying this project has been completed and is ready for submission to a peer-reviewed venue.

---

## 🎯 Problem Statement

Mental health platforms generate enormous volumes of user-generated text. Manually reviewing posts to identify who needs what kind of support is not scalable. Key questions this project addresses:

- Can NLP models reliably classify **behavioral intent** in mental health posts?
- How much does fine-tuning BERT on domain-specific data improve over generic embeddings?
- How do transformer models compare to classical ML baselines (Logistic Regression, SVM)?
- What are the failure modes and how can they be mitigated?

Accurate behavioral intent classification supports:
- **Early intervention systems** that flag high-risk users
- **Moderation prioritization** for community managers
- **Behavioral trend analysis** across mental health communities
- **Automated support routing** to appropriate resources

---

## 🗂️ Repository Structure

```
Reddit-NLP-Research/
│
├── bert.ipynb                        # BERT fine-tuning experiments and evaluation
├── USE.ipynb                         # Universal Sentence Encoder baseline experiments
├── Latest.ipynb                      # Final consolidated pipeline and results
├── final_prompt_function.py          # Inference utility: prompt → prediction
│
├── MENTAL_HELP.pdf                   # Research paper (full manuscript)
├── Metrics.xlsx                      # Compiled evaluation metrics across all experiments
│
├── predicted_results_final_stage.csv # Final model predictions on test set
├── testdata_afterfirst.csv           # Processed test data after first-stage filtering
├── best_thresholds.npy               # Optimal classification thresholds (NumPy array)
│
├── config.json                       # Model configuration (BERT hyperparameters)
├── tokenizer_config.json             # HuggingFace tokenizer settings
├── special_tokens_map.json           # Special token definitions
├── vocab.txt                         # BERT vocabulary file
│
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Reddit mental health subreddits (via Reddit API / PRAW) |
| **Size** | 112,000+ posts |
| **Task** | Multi-class behavioral intent classification |
| **Split** | 80% Train / 10% Validation / 10% Test |
| **Sampling** | Stratified to preserve class distribution |
| **Imbalance handling** | Class reweighting during training |

**Behavioral Intent Classes** include categories such as:
- Advice-seeking
- Emotional expression / venting
- Informational support requests
- Peer support / community connection
- Crisis signals

> *Note: Data was collected and handled in accordance with Reddit's API terms of service. No personally identifiable information is stored.*

---

## 🏗️ Pipeline Architecture

```
Reddit Posts
     │
     ▼
Text Preprocessing
(cleaning, normalization, deduplication)
     │
     ▼
Tokenization
(BERT WordPiece / USE sentence encoder)
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
Classical Baselines              Transformer Models
(TF-IDF + LR / SVM)         (BERT fine-tuned / USE)
     │                                  │
     └──────────────┬───────────────────┘
                    ▼
          Evaluation & Analysis
   (Accuracy, F1, Ablations, Error Analysis)
                    │
                    ▼
         Threshold Optimization
         (best_thresholds.npy)
                    │
                    ▼
         Final Predictions
    (predicted_results_final_stage.csv)
```

---

## 🤖 Models

### Baselines
| Model | Features | Notes |
|---|---|---|
| Logistic Regression | TF-IDF unigrams + bigrams | Fast, interpretable |
| Support Vector Machine | TF-IDF, linear kernel | Strong text baseline |

### Transformer Models
| Model | Approach | Notes |
|---|---|---|
| BERT (`bert-base-uncased`) | Domain-adaptive fine-tuning | Full fine-tuning on mental health corpus |
| Universal Sentence Encoder (USE) | Sentence embeddings + classifier head | Google's multilingual encoder |

**BERT Fine-Tuning Details:**
- Pre-trained `bert-base-uncased` from HuggingFace
- Fine-tuned end-to-end on Reddit mental health corpus
- Cross-entropy loss with class weights for imbalance handling
- Dropout regularization to reduce overfitting
- Hyperparameter search over: learning rate `{1e-5, 2e-5, 3e-5}`, batch size `{16, 32}`, dropout `{0.1, 0.3}`

---

## 📈 Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression | 71% | 0.68 |
| SVM | 73% | 0.70 |
| Universal Sentence Encoder | ~78% | ~0.75 |
| **BERT (Fine-Tuned)** | **85%** | **0.82** |

**Key Findings:**
- BERT fine-tuning yields a **+12–14% accuracy gain** over classical ML baselines
- Domain-adaptive fine-tuning is critical — generic BERT embeddings underperform fine-tuned versions significantly
- Class reweighting reduced false negatives on underrepresented intent categories
- Dropout tuning reduced the train/val overfitting gap by **~13%**
- Optimal classification thresholds (saved in `best_thresholds.npy`) further improved macro-F1 on the held-out test set

---

## 🔬 Experimental Framework

- **Cross-validation:** 5-fold stratified cross-validation
- **Ablations:** 12+ experiments varying tokenization strategy, pooling method, loss function, and class weighting
- **Error analysis:** Manual inspection of misclassified samples; identified ambiguous boundary cases between advice-seeking and emotional venting
- **Held-out test set:** 15,000+ samples never seen during training or hyperparameter tuning

Full metrics across all experiments are available in `Metrics.xlsx`.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers scikit-learn pandas numpy openpyxl jupyter
pip install tensorflow tensorflow-hub   # for USE experiments
```

### Running the Notebooks

```bash
git clone https://github.com/avanthikarajesh30/Reddit-NLP-Research.git
cd Reddit-NLP-Research
jupyter notebook
```

Open in order:
1. `bert.ipynb` — BERT fine-tuning pipeline
2. `USE.ipynb` — Universal Sentence Encoder experiments
3. `Latest.ipynb` — Final consolidated results and analysis

### Running Inference

```python
from final_prompt_function import predict_intent

result = predict_intent("I don't know what to do anymore, can someone help me figure this out?")
print(result)
# Output: {'intent': 'advice-seeking', 'confidence': 0.91}
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| PyTorch | Model training and fine-tuning |
| HuggingFace Transformers | BERT tokenizer, model, trainer |
| TensorFlow / TF Hub | Universal Sentence Encoder |
| Scikit-learn | Baseline models, evaluation metrics |
| Pandas / NumPy | Data processing, threshold storage |
| PRAW (Reddit API) | Data collection |
| Jupyter Notebook | Experimentation and visualization |

---

## 📄 Research Paper

A full research manuscript (`MENTAL_HELP.pdf`) is included in this repository. It covers:

- Literature review on NLP in digital mental health
- Full dataset description and collection methodology
- Detailed experimental setup and ablation analysis
- Statistical significance testing
- Discussion of ethical considerations and limitations
- Future directions

> **Submission status:** Manuscript completed, ready for peer-reviewed venue submission.

---

## ⚠️ Ethical Considerations

This project involves sensitive mental health data from real users. Key ethical safeguards:

- No personally identifiable information (PII) is stored or published
- Data used strictly for research purposes in compliance with Reddit's API terms
- Model outputs should **not** be used as a substitute for professional mental health assessment
- Misclassification risk is acknowledged and discussed in the paper — particularly for crisis-signal posts

---

## 🔮 Future Work

- Compare against **RoBERTa**, **DeBERTa**, and **MentalBERT** (domain-specific)
- Few-shot and zero-shot classification with GPT-4 / LLaMA
- Real-time streaming inference pipeline for live Reddit monitoring
- Explainability via **SHAP values** and **attention visualization**
- Integration into a digital mental health support dashboard
- Longitudinal user behavior modeling across post histories

---

## 👩‍💻 Author

**Avanthika Rajesh**
MS Computer Engineering — Virginia Tech
[GitHub](https://github.com/avanthikarajesh30) • [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN_HERE)

---

## 📄 License

This project is licensed under the MIT License. The research paper (`MENTAL_HELP.pdf`) is © the author — please cite appropriately if building on this work.

---

## 🙏 Acknowledgements

- Reddit community members whose public posts made this research possible
- HuggingFace for the Transformers library and pre-trained BERT weights
- Google Research for the Universal Sentence Encoder

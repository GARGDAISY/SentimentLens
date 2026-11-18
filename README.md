# SentimentLens – Multilingual Review Sentiment Analyzer

A Python-based NLP application built with **Streamlit** that analyzes customer reviews and classifies them as positive, neutral, or negative.

SentimentLens supports multiple file formats and pasted review text. It provides sentiment distribution, language detection, duplicate-review signals, and downloadable PDF and CSV reports.

## ✨ Features

- **Multiple Input Formats:** Upload PDF, DOCX, XLSX, XLS, CSV, and TXT files or paste review text directly.
- **Sentiment Classification:** Classify reviews as positive, neutral, or negative and generate an overall sentiment verdict.
- **Multilingual Support:** Detect English, Mandarin Chinese, Hindi, Spanish, Arabic, French, and Bengali.
- **Hinglish Support:** Normalize Hinglish text using custom dictionary-based rules and negation handling.
- **Offline Translation:** Translate supported non-English languages into English using Argos Translate.
- **Emoji-Aware Sentiment:** Handle emoji-only and mixed text-and-emoji reviews separately.
- **Duplicate Detection:** Identify duplicate and similar reviews using TF-IDF and sparse matrix operations.
- **Exportable Reports:** Generate downloadable PDF and CSV reports.
- **Review-Level Analysis:** View original reviews, English translations, detected languages, and sentiment classifications.

## 🌍 Supported Language Workflow

SentimentLens uses different processing methods depending on the detected language:

| Language | Processing Method |
|---|---|
| English | Direct sentiment classification |
| Hinglish | Custom normalization and negation rules |
| Hindi | Local Argos translation to English |
| Mandarin Chinese | Local Argos translation to English |
| Spanish | Local Argos translation to English |
| Arabic | Local Argos translation to English |
| French | Local Argos translation to English |
| Bengali | Local Argos translation to English |

**Translation details:**
- Argos Translate models must be installed explicitly.
- Translation models are not downloaded automatically when the application starts.
- Missing translation models do not crash the application.
- The application displays a warning when a required translation model is unavailable.

Language detection may be uncertain for short text, mixed scripts, slang, dialects, and transliterated language. Translation quality depends on the installed model and language variety.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| NLP & Text Processing | TF-IDF, Text Preprocessing |
| Sentiment Analysis | Machine Learning |
| Language Detection | Language Detection |
| Translation | Argos Translate |
| Data Processing | Pandas, NumPy |
| Similarity Detection | Scikit-learn, Sparse Matrices |
| Reporting | PDF and CSV |
| Testing | Pytest |

## 🚀 Installation & Setup

### Prerequisites

- Python 3.12
- VS Code
- Git (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/GARGDAISY/SentimentLens.git
cd SentimentLens
```

### 2. Create a Virtual Environment

Open PowerShell in the project directory:

```powershell
py -3.12 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run the following command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 🌐 Install Offline Translation Models

This is a one-time setup step that requires internet access to retrieve the Argos package index and download the selected translation models.

```powershell
.\.venv\Scripts\python.exe scripts\install_argos_models.py
```

The setup script installs six translation models:

- Mandarin Chinese → English
- Hindi → English
- Spanish → English
- Arabic → English
- French → English
- Bengali → English

The approximate total download size is 650 MB. Actual sizes may vary depending on package versions.

Translation models are stored in Argos Translate's user-level model directory and are not bundled with the project.

To skip Hindi translation:

```powershell
.\.venv\Scripts\python.exe scripts\install_argos_models.py --without-hindi
```

Restart the Streamlit application after installing the models so the language-support status refreshes.

## ▶️ Run the Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will provide a local URL to open the application in your browser.

## 🧪 Run Tests

Execute the test suite using:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 📊 Training Data & Model Limitations

- `data/train_seed.csv` contains a small synthetic dataset used to make the project runnable out of the box.
- Evaluation scores obtained from this synthetic dataset should not be treated as evidence of real-world model accuracy.
- Meaningful evaluation requires a separate, human-labelled review dataset and a held-out test set.
- Accuracy and macro-F1 should be reported using the held-out test set.
- The base sentiment model is English-focused.
- Non-English reviews should be successfully translated before classification.
- If a translation model is unavailable, the application warns that the original text was passed to an English-focused classifier.
- An optional XLM-RoBERTa fallback is not installed by default and requires an additional model download.

### Emoji Handling

Emoji sentiment uses a small, transparent lexicon.

- Emoji-only and mixed text-and-emoji reviews are handled separately.
- Mixed reviews use a modest emoji contribution.
- Context-dependent emojis such as 😂, 🔥, and 💀 are not assigned fixed sentiment values.
- Sarcasm and context-dependent emoji meanings may affect classification.

### Report Limitations

PDF reports use built-in fonts. Non-Latin characters may appear as `?` in PDF review samples.

CSV exports preserve Unicode text.

## 📁 Project Structure

```text
SentimentLens/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── src/
│   ├── analysis.py
│   ├── emoji_sentiment.py
│   ├── extractors.py
│   ├── hinglish.py
│   ├── language.py
│   ├── model.py
│   ├── multilingual.py
│   ├── preprocess.py
│   ├── report.py
│   └── similarity.py
│
├── scripts/
│   ├── install_argos_models.py
│   └── make_seed_data.py
│
├── setup_windows.ps1
│
├── data/
├── models/
│
└── tests/
```

## 📌 Future Improvements

- Improve sentiment classification accuracy using larger labelled datasets.
- Expand language and dialect support.
- Improve context-aware emoji and sarcasm detection.
- Enhance multilingual sentiment evaluation.

---

**Developed by Nandini Garg**
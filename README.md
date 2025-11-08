# Language Translator: English to Indic Languages (Kashmiri & Others)

## Project Title and Description

This project implements a state-of-the-art machine translation system for translating English text to various Indic languages, with a primary focus on Kashmiri. The system leverages AI4Bharat's IndicTrans2 model, a powerful multilingual translation model optimized for Indian languages, to provide accurate and context-aware translations.

### Key Features

- **English to Kashmiri Translation**: High-quality translation from English to Kashmiri (Devanagari script)
- **English to Other Indic Languages**: Support for translation to multiple Indic languages
- **Batch Translation**: Efficient batch processing for multiple sentences
- **Quantization Support**: 4-bit and 8-bit quantization for memory-efficient inference
- **Interactive Interface**: Gradio-based web interface for easy translation
- **Entity Preservation**: Maintains proper names, dates, and technical terms in translations

### Project Objectives

1. Develop an AI-powered translation system for converting English text into Kashmiri while preserving meaning and context
2. Enhance translation quality through preprocessing and postprocessing techniques
3. Optimize the model for efficiency using quantization techniques (4-bit and 8-bit) to support lower-memory devices
4. Ensure script accuracy by handling Kashmiri script-specific nuances effectively
5. Improve entity recognition and formatting to maintain proper names, dates, and technical terms
6. Provide an accessible interface for translation tasks

---

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Dataset Information](#dataset-information)
- [Directory Structure](#directory-structure)
- [Usage Examples](#usage-examples)
- [Technologies Used](#technologies-used)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) for faster inference
- Git
- pip package manager

> **⚠️ Note**: Running this model requires high computational power. It is **strongly recommended** to use **Google Colab** for execution, especially if you don't have a GPU-enabled machine.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Language-Translator
```

### Step 2: Create and Activate a Virtual Environment

#### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Additionally, download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Step 4: Install IndicTrans2 and IndicTransToolkit

The project requires the IndicTrans2 repository and IndicTransToolkit. Run the following commands:

```bash
# Clone IndicTrans2 repository
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/huggingface_interface

# Clone and install IndicTransToolkit
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install --editable ./
cd ../..
```

### Step 5: Download Models

The models will be automatically downloaded from Hugging Face when you run the notebooks. The following models are used:

- **English to Indic**: `ai4bharat/indictrans2-en-indic-1B`
- **Indic to English**: `ai4bharat/indictrans2-indic-en-1B` (optional)
- **Indic to Indic**: `ai4bharat/indictrans2-indic-indic-1B` (optional)

> **⚠️ Security Note**: For accessing Hugging Face models, you may need to authenticate using a Hugging Face token. You can obtain a token from [Hugging Face](https://huggingface.co/settings/tokens).
> 
> **IMPORTANT**: Never commit your Hugging Face token to the repository! Always use secure methods like:
> - `getpass()` function in Python (as shown in the notebooks)
> - Environment variables
> - Google Colab secrets (for Colab notebooks)
> - Secret management tools for production environments

### Step 6: Download Datasets (Optional)

If you plan to train or evaluate the model, you may need to download datasets:

- **FLORES Dataset**: Available at [Hugging Face Datasets - FLORES](https://huggingface.co/datasets/facebook/flores)
- Custom CSV datasets with `eng_Latn` and `doi_Deva` columns for English-Kashmiri pairs

---

## Dataset Information

### Primary Dataset

The project can work with custom datasets or the FLORES dataset for evaluation:

- **FLORES Dataset**
  - **Source**: [Hugging Face - facebook/flores](https://huggingface.co/datasets/facebook/flores)
  - **Description**: FLORES is a multilingual evaluation benchmark covering multiple languages including Kashmiri
  - **Format**: Parallel sentences in English and target languages
  - **Usage**: Used for evaluation and BLEU score calculation

### Dataset Format

For training or fine-tuning, datasets should be in CSV format with the following columns:

- `eng_Latn`: English text (Latin script)
- `doi_Deva`: Kashmiri/Dogri text (Devanagari script)

Example CSV structure:
```csv
eng_Latn,doi_Deva
"Hello, how are you?","नमस्ते, तुहाड़ा क्या हाल है?"
"What is your name?","तुहाड़ा नाव की है?"
```

### Dataset Preprocessing

The project includes functions to:
- Load data from CSV files
- Split data into train, validation, and test sets (80/10/10 split)
- Clean and normalize text data
- Handle missing values and empty strings
- Save processed data in text file format for model training

---

## Directory Structure

```
Language-Translator/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── kashmiri_mt_model.ipynb           # Kashmiri translation notebook with Gradio interface
├── LanguageTranslator.ipynb           # Main translation notebook (comprehensive)
├── LanguageTranslatorEngToOther.ipynb # English to other Indic languages notebook
│
└── (Generated during execution)
    ├── IndicTrans2/                   # Cloned IndicTrans2 repository
    ├── IndicTransToolkit/             # Cloned IndicTransToolkit
    └── data/                          # Processed dataset files (if using custom datasets)
        ├── train.en
        ├── train.doi
        ├── dev.en
        ├── dev.doi
        ├── test.en
        └── test.doi
```

### File Descriptions

- **`kashmiri_mt_model.ipynb`**: Simplified notebook focused on English to Kashmiri translation with interactive Gradio interface
- **`LanguageTranslator.ipynb`**: Comprehensive notebook with full translation pipeline, evaluation metrics, and multiple language support
- **`LanguageTranslatorEngToOther.ipynb`**: Notebook for English to various Indic languages translation

---

## Usage Examples

### Basic Translation (Python)

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Initialize model
ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, trust_remote_code=True)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Initialize processor
ip = IndicProcessor(inference=True)

# Translate
src_lang, tgt_lang = "eng_Latn", "doi_Deva"
input_sentences = ["Hello, how are you?", "What is your name?"]

# Preprocess
batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

# Tokenize
inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    generated_tokens = model.generate(**inputs, max_length=256, num_beams=5)

# Decode
translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
translations = ip.postprocess_batch(translations, lang=tgt_lang)

print(translations)
```

### Interactive Translation (Notebook)

Open `kashmiri_mt_model.ipynb` in Google Colab and run all cells. The notebook includes:

1. Setup and installation
2. Model initialization
3. Interactive translation interface
4. Gradio web interface

### Input and Output Examples

#### Example 1: Simple Translation
**Input (English):**
```
Hello, how are you?
```

**Output (Kashmiri):**
```
नमस्ते, तुहाड़ा क्या हाल है?
```

#### Example 2: Question Translation
**Input (English):**
```
What is your name?
```

**Output (Kashmiri):**
```
तुहाड़ा नाव की है?
```

#### Example 3: Complex Sentence
**Input (English):**
```
I am learning machine translation.
```

**Output (Kashmiri):**
```
मैं मशीन अनुवाद सीख रहा हूं।
```

#### Example 4: Word Translation
**Input (English):**
```
told
```

**Output (Kashmiri):**
```
दस्सेआ
```

#### Example 5: Phrase Translation
**Input (English):**
```
ancestors
```

**Output (Kashmiri):**
```
बापस दे पुरखे
```

---

## Technologies Used

### Programming Languages
- **Python 3.8+**: Primary programming language

### Core Libraries and Frameworks

#### Deep Learning & NLP
- **PyTorch** (`torch`): Deep learning framework for model execution
- **Hugging Face Transformers** (`transformers>=4.33.2`): Pre-trained model loading and inference
- **Hugging Face Datasets** (`datasets`): Dataset handling and processing
- **Hugging Face Accelerate** (`accelerate`): Optimization for GPU inference

#### NLP Processing
- **NLTK** (`nltk`): Natural language processing toolkit for sentence segmentation
- **SentencePiece** (`sentencepiece`): Tokenization and subword processing
- **Moses Tokenizer** (`mosestokenizer`): Advanced tokenization for better linguistic structuring
- **Sacremoses** (`sacremoses`): Python port of Moses tokenizer

#### Machine Learning & Data
- **NumPy** (`numpy`): Numerical computing
- **Pandas** (`pandas`): Data manipulation and analysis
- **Scikit-learn** (`sklearn`): Data splitting and evaluation utilities
- **SciPy** (`scipy`): Scientific computing

#### Optimization
- **BitsAndBytes** (`bitsandbytes`): Quantization support (4-bit and 8-bit) for efficient inference

#### Utilities
- **Regex** (`regex`): Advanced regular expressions
- **Mock** (`mock`): Testing utilities

#### Custom Tools
- **IndicTransToolkit**: Custom toolkit for preprocessing and postprocessing Indic language text
  - Entity recognition and preservation
  - Script normalization
  - Tokenization and detokenization

#### Interface
- **Gradio** (`gradio`): Web interface for interactive translation (optional)

### Model Architecture
- **IndicTrans2**: Transformer-based sequence-to-sequence model
  - Model: `ai4bharat/indictrans2-en-indic-1B` (1 billion parameters)
  - Architecture: Encoder-Decoder Transformer
  - Pre-trained on large-scale multilingual corpora

---

## Model Information

### Pre-trained Models

The project uses AI4Bharat's IndicTrans2 models available on Hugging Face:

1. **English to Indic** (`ai4bharat/indictrans2-en-indic-1B`)
   - Translates from English to multiple Indic languages
   - Model size: ~1B parameters
   - Supports 20+ Indic languages including Kashmiri

2. **Indic to English** (`ai4bharat/indictrans2-indic-en-1B`) - Optional
   - Translates from Indic languages to English
   - Useful for bidirectional translation

3. **Indic to Indic** (`ai4bharat/indictrans2-indic-indic-1B`) - Optional
   - Translates between different Indic languages
   - Supports direct translation without English as intermediary

### Model Features

- **Multilingual Support**: Handles multiple Indic languages
- **Script Handling**: Proper handling of Devanagari and other Indic scripts
- **Entity Preservation**: Maintains proper names, dates, and technical terms
- **Context Awareness**: Context-aware translations preserving meaning
- **Quantization Support**: Supports 4-bit and 8-bit quantization for memory efficiency

### Language Codes

The project uses language codes in the format `{language}_{script}`:

- `eng_Latn`: English (Latin script)
- `doi_Deva`: Dogri/Kashmiri (Devanagari script)
- `hin_Deva`: Hindi (Devanagari script)
- `mar_Deva`: Marathi (Devanagari script)
- `kas_Deva`: Kashmiri (Devanagari script)

---

## Project Architecture

```
┌─────────────────┐
│  Input Text     │
│  (English)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Tokenization │
│  - Entity       │
│    Recognition  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IndicTrans2    │
│  Model          │
│  (Encoder-      │
│   Decoder)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Postprocessing │
│  - Detokenize   │
│  - Entity       │
│    Replacement  │
│  - Formatting   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Text    │
│  (Kashmiri)     │
└─────────────────┘
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgments

- **AI4Bharat**: For developing and open-sourcing the IndicTrans2 model
- **Hugging Face**: For providing the model hosting and Transformers library
- **IndicTransToolkit Developers**: For the preprocessing and postprocessing toolkit
- **Facebook Research**: For the FLORES evaluation dataset
- **Open Source Community**: For various NLP libraries and tools

### References

- [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2)
- [IndicTrans2 Hugging Face](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
- [IndicTransToolkit GitHub](https://github.com/VarunGumma/IndicTransToolkit)
- [FLORES Dataset](https://huggingface.co/datasets/facebook/flores)

---

## License

This project is open source and available under the MIT License (or as specified in the repository).

---

## Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.

---

## Additional Notes

- **GPU Recommendation**: For best performance, use a CUDA-enabled GPU. The model can run on CPU but will be significantly slower.
- **Memory Requirements**: The full model requires approximately 4-8 GB of GPU memory. Use quantization for lower memory requirements.
- **Google Colab**: Recommended platform for running the notebooks without local GPU setup.
- **Hugging Face Token**: May be required for accessing certain models. Obtain from [Hugging Face Settings](https://huggingface.co/settings/tokens).

---
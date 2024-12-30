# Skirpsi-Extractive-summarization
# Text Summarization Application

This project is a text summarization application built using Streamlit, NLTK, and TensorFlow. It allows users to input a document and generates a concise summary using a trained model. The application also supports evaluation of the generated summary using ROUGE scores and provides functionality for batch processing via CSV file uploads.

## Features
- **Interactive Text Summarization**: Enter a document to generate a summary interactively.
- **ROUGE Evaluation**: Compare generated summaries with reference texts and calculate ROUGE scores.
- **Batch Processing**: Upload a CSV file to process multiple documents and generate summaries.
- **Customizable Summarization**: Uses a trained model (e.g., `M1.h5`) with Word2Vec embeddings for feature extraction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-summarization-app.git
   cd text-summarization-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```bash
   python -m nltk.downloader punkt
   ```

## Usage

### Running the Application
Run the Streamlit app locally:
```bash
streamlit run app.py
```

### Input Options
Input Must in Bahasa Indonesia
1. **Text Input**:
   - Enter the document to summarize in the provided text area.
   - Optionally, provide a reference summary for ROUGE evaluation.

2. **CSV Upload**:
   - Upload a CSV file with a column named `Sum_text` for documents to summarize.
   - (Optional) Include a column named `summary_text` for reference summaries.

### Output
- Generated summaries are displayed on the interface.
- ROUGE scores are computed and shown for individual and batch processing.

## Example
### Input Document
```
Alif pergi ke Lampung. Disana dia menemukan bunga Rafflesia.
```
### Generated Summary
```
Alif pergi ke Lampung
```

### ROUGE Scores
- **Rouge-1 Precision**: 0.3287
- **Rouge-1 Recall**: 0.6045
- **Rouge-1 F1 Score**: 0.4207
- **Rouge-2 Precision**: 0.2159
- **Rouge-2 Recall**: 0.4351
- **Rouge-2 F1 Score**: 0.2841

## File Structure
```
project-directory/
 ├── Extractive app/
  ├── app.py                          # Main application script
  ├── model_loader.py                 # Model loading utility
  ├── summary_generator.py            # Summary generation logic
  ├── rouge_evaluator.py              # ROUGE evaluation logic
  ├── text_processor.py               # Text preprocessing utilities
  ├── requirements.txt                # Python dependencies
├── README.md                         # Documentation (this file)
├── train_extractive_summary.ipynb    # training file
├── Filter dataset                    # Dataset used to training ( Indonesian News)
```

## Dependencies
- **Python**: 3.8+
- **Streamlit**: Interactive web application framework.
- **TensorFlow/Keras**: Deep learning framework for model loading.
- **NLTK**: Natural Language Toolkit for tokenization.
- **ROUGE**: Evaluation metric for summarization.
- **Pandas**: Data manipulation and analysis.

## License



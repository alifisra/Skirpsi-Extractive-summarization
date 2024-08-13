import streamlit as st
import pandas as pd
import nltk
import re
from model_loader import ModelLoader
from text_processor import TextProcessor
from summary_generator import SummaryGenerator
from rouge_evaluator import RougeEvaluator

# Load NLTK data
nltk.download('punkt')

# Define the base path
base_path = "D:\Skripsi Project\model"

# Initialize components
model_loader = ModelLoader(base_path)
w2v_vocab = model_loader.get_w2v_vocab()
text_processor = TextProcessor(w2v_vocab, max_len=1319)
summary_generator = SummaryGenerator(model_loader.M1, text_processor)
rouge_evaluator = RougeEvaluator()
#pattern = r'\\xa0|\b(?:Jakarta\s*,\s*CNN\s*Indonesia\s*-\s*-|Berlin\s*\(\s*ANTARA\s*News\s*\)\s*–|Vientiane\s*\(\s*ANTARA\s*News\s*\)\s*-\s*|\w+\.com\s*-\s*|\w+\.id\s*:\s*|KOREA SELATAN\s*–)\b|simak juga\s*:|Jakarta \( ANTARA News \) -\s*|Moskow ( ANTARA News ) -'

# Streamlit UI
st.title("Text Summarization App")

# Choose input method
input_method = st.radio("Jenis Input:", ('Manual Text Input', 'Upload CSV File'))

if input_method == 'Manual Text Input':
    # Text input for user document
    user_input = st.text_area("masukkan dokumen yang ingin diringkas")

    # Reference text input
    #reference_text = st.text_area("masukkan ringkasan referensi (optional):")

    if st.button("Generate Summary"):
        if user_input:
            # Generate summary for user input
            user_summary, top_indices= summary_generator.generate_summary(user_input)
            st.subheader("Generated Summary:")
        #    updated_summary = re.sub(pattern, '', user_summary, flags=re.IGNORECASE)
            st.write(user_summary)


            # If reference text is provided, evaluate ROUGE scores
            #if reference_text:
            #    st.subheader("Reference Summary:")
            #    st.write(reference_text)
                # Calculate ROUGE scores
            #    rouge_scores = rouge_evaluator.evaluate([user_summary], [reference_text])
            #   st.subheader("ROUGE Scores:")
            #    rouge_evaluator.display_scores(rouge_scores)
        else:
            st.warning("masukkan text yang akan diringkas.")

elif input_method == 'Upload CSV File':
    # Option to upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load and display the uploaded CSV file
        test_df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.write(test_df)

        if 'news_text' in test_df.columns:
            # Process the documents and generate summaries
            test_sentences = test_df['news_text'].tolist()
            summaries = []

            for doc in test_sentences:
                summary,top_indices = summary_generator.generate_summary(doc)
                summaries.append(summary)  # Append each generated summary to the list

            test_df['generated_summary'] = summaries

            # Function to replace matched strings with an empty string
            #test_df['generated_summary'] = test_df['generated_summary'].apply(lambda x: re.sub(pattern, '', x, flags=re.IGNORECASE))

            # If reference summaries are available, calculate ROUGE scores
            if 'summary_text' in test_df.columns:
                # Display the dataframe with the generated summaries
                st.write("CSV file with generated summaries:")
                st.write(test_df[['news_text', 'generated_summary', 'summary_text']])
                non_empty_df = test_df[test_df['generated_summary'] != '']
                rouge_scores = rouge_evaluator.evaluate(non_empty_df['generated_summary'].tolist(), non_empty_df['summary_text'].tolist())
                st.write("ROUGE scores:")
                st.write("Average ROUGE scores:")
                rouge_evaluator.display_scores(rouge_scores)
                # Calculate individual ROUGE scores
                individual_rouge_scores = rouge_evaluator.evaluate_individual(non_empty_df['generated_summary'].tolist(), non_empty_df['summary_text'].tolist())
                flattened_scores = []
                for score in individual_rouge_scores:
                    flattened = {f"{metric}_{key}": value for metric, results in score.items() for key, value in results.items()}
                    flattened_scores.append(flattened)
                individual_rouge_df = pd.DataFrame(flattened_scores)
                test_df = pd.concat([test_df.reset_index(drop=True), individual_rouge_df.reset_index(drop=True)], axis=1)
                st.write("Individual ROUGE scores:")
                st.write(test_df)
            else:
                st.write("CSV file with generated summaries:")
                st.write(test_df[['news_text', 'generated_summary']])
        else:
            st.write('Error: tidak ditemukan kolom "news_text", text yag akan diringkas.')

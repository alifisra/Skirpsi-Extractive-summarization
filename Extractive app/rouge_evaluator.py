from rouge import Rouge
import streamlit as st

class RougeEvaluator:
    def __init__(self):
        self.rouge = Rouge()

    def evaluate_individual(self, generated_summaries, reference_summaries):
        return self.rouge.get_scores(generated_summaries, reference_summaries, avg=False)
    def evaluate(self, generated_summaries, reference_summaries):
        rouge_scores = self.rouge.get_scores(generated_summaries, reference_summaries, avg=True)
        return rouge_scores

    def display_scores(self, rouge_scores):
        rouge_1_scores = rouge_scores['rouge-1']
        rouge_2_scores = rouge_scores['rouge-2']

        st.write("**Rouge-1 Scores:**")
        st.write("Precision:", rouge_1_scores['p'])
        st.write("Recall:", rouge_1_scores['r'])
        st.write("F1 Score:", rouge_1_scores['f'])

        st.write("**Rouge-2 Scores:**")
        st.write("Precision:", rouge_2_scores['p'])
        st.write("Recall:", rouge_2_scores['r'])
        st.write("F1 Score:", rouge_2_scores['f'])

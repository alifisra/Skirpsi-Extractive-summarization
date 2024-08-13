import numpy as np
from nltk.tokenize import sent_tokenize

class SummaryGenerator:
    def __init__(self, model, text_processor):
        self.model = model
        self.text_processor = text_processor

    def generate_summary(self, document, num_top_sentences=3):
        sentences = sent_tokenize(document)  # Split document into sentences
        sequences = self.text_processor.tokenize_and_convert_to_sequences(sentences)
        padded_sequences = self.text_processor.pad_sequences(sequences)

        # Make predictions
        if len(padded_sequences) == 0:
            return ''

        predictions = self.model.predict(padded_sequences)
        
        # Handle cases where there are fewer than 4 sentences
        num_sentences = len(sentences)
        num_top_sentences = max(1, int(np.round(0.3 * num_sentences)))  # Ensure at least one sentence is included

        # Adjusting the sorting mechanism
        predictions = np.array(predictions).flatten()

        # Ensure the top indices are unique and handle repetitive predictions
        top_indices = np.argpartition(predictions, -num_top_sentences)[-num_top_sentences:]
        top_indices = top_indices[np.argsort(-predictions[top_indices])]
        top_indices = list(set([int(idx / 75) for idx in top_indices]))  # Ensure indices are unique

        if len(top_indices) < num_top_sentences:
            # If unique indices are fewer than required, add additional unique indices
            remaining_indices = list(set(range(len(sentences))) - set(top_indices))
            np.random.shuffle(remaining_indices)
            top_indices.extend(remaining_indices[:num_top_sentences - len(top_indices)])

        # Debugging: Print the top indices
        top_indices.sort()

        # Generate summary by selecting top sentences
        summary = ' '.join([sentences[idx] for idx in top_indices if idx < num_sentences])
        return summary, top_indices

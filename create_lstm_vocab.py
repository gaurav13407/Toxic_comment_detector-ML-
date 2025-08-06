# Script to save LSTM preprocessing data for production use
# Run this after training the LSTM model to save vocabulary and preprocessing info

import pickle
import os
import pandas as pd
from collections import Counter

# Make NLTK optional
try:
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
    except:
        print("Warning: Could not download NLTK punkt tokenizer")
except ImportError:
    print("Warning: NLTK not available, using simple tokenization")
    NLTK_AVAILABLE = False

def create_vocabulary_from_data():
    """Create vocabulary from the training data"""
    try:
        # Load the dataset
        df = pd.read_csv('data/processed/final_training_dataset.csv')
        texts = df['comment'].astype(str).tolist()
        
        print(f"Processing {len(texts)} texts to create vocabulary...")
        
        # Tokenize using NLTK if available, otherwise simple split
        tokenized_words = []
        for i, text in enumerate(texts):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(texts)} texts")
            try:
                if NLTK_AVAILABLE:
                    tokens = word_tokenize(text.lower())
                else:
                    tokens = text.lower().split()
                tokenized_words.append(tokens)
            except:
                # Fallback to simple split
                tokens = text.lower().split()
                tokenized_words.append(tokens)
        
        # Build the vocabulary
        all_tokens = [token for sublist in tokenized_words for token in sublist]
        vocab = Counter(all_tokens)
        vocab_size = len(vocab)
        most_common = vocab.most_common(vocab_size - 2)
        
        word2indx = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(most_common):
            word2indx[word] = i + 2
        
        print(f"Created vocabulary with {len(word2indx)} words")
        
        # Save preprocessing info
        preprocessing_info = {
            'word2indx': word2indx,
            'max_length': 100,
            'max_vocab_size': 2000,
            'vocab_size': len(word2indx),
            'creation_date': str(pd.Timestamp.now()),
            'num_texts_processed': len(texts)
        }
        
        # Create output directory if it doesn't exist
        os.makedirs('output/models', exist_ok=True)
        
        # Save preprocessing info for LSTM model
        with open('output/models/lstm_preprocessing_info.pkl', 'wb') as f:
            pickle.dump(preprocessing_info, f)
        
        print("✅ LSTM preprocessing info saved to: output/models/lstm_preprocessing_info.pkl")
        return preprocessing_info
        
    except Exception as e:
        print(f"❌ Error creating vocabulary: {e}")
        return None

if __name__ == "__main__":
    # Change to the correct directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    print("Creating vocabulary from training data...")
    vocab_info = create_vocabulary_from_data()
    
    if vocab_info:
        print(f"✅ Vocabulary created successfully!")
        print(f"📊 Vocabulary size: {vocab_info['vocab_size']}")
        print(f"📝 Max length: {vocab_info['max_length']}")
        print(f"🔢 Max vocab size: {vocab_info['max_vocab_size']}")
    else:
        print("❌ Failed to create vocabulary")

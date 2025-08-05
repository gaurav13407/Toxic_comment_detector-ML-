import joblib
import numpy as np
import re
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
models_dir = os.path.join(project_root, 'output', 'models')

## Load model and embedder
model = joblib.load(os.path.join(models_dir, 'toxic_comment_classifier_random_forest_fast.joblib'))

## Load Word2Vec Model
w2v_model = Word2Vec.load(os.path.join(models_dir, 'word2vec_model.model'))

## Configuration
VECTOR_SIZE = 300

def preprocess_text(text):
    """Clean and tokenize text for Word2Vec training."""
    if not text or pd.isna(text):
        return []
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    
    # Tokenize using gensim's simple_preprocess
    tokens = simple_preprocess(text, deacc=True, min_len=2, max_len=15)
    
    return tokens

def get_text_embedding(tokens, model):
    """Generate embedding for a text by averaging word vectors."""
    if not tokens:
        return np.zeros(VECTOR_SIZE)
    
    # Get word vectors for tokens that exist in vocabulary
    word_vectors = []
    for token in tokens:
        if token in model.wv:
            word_vectors.append(model.wv[token])
    
    if not word_vectors:
        return np.zeros(VECTOR_SIZE)
    
    # Average the word vectors
    return np.mean(word_vectors, axis=0)

def get_embedding(texts):
    """
    Convert texts to Word2Vec embeddings
    
    Args:
        texts: Single text string or list of text strings
    
    Returns:
        numpy array of embeddings
    """
    # Handle single text input
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = []
    
    for text in texts:
        # Preprocess text
        tokens = preprocess_text(text)
        
        # Get embedding
        embedding = get_text_embedding(tokens, w2v_model)
        embeddings.append(embedding)
    
    return np.array(embeddings)

def predict_toxicity(comment, sensitivity=0.3):
    """
    Predict if a comment is toxic with adjustable sensitivity
    
    Args:
        comment (str): The comment to analyze
        sensitivity (float): Lower values = more sensitive (0.1-0.5 recommended)
    
    Returns:
        dict: Prediction results with toxicity labels and scores
    """
    # Get embedding
    embedding = get_embedding(comment)
    
    # Make prediction
    prediction = model.predict(embedding)[0]
    probabilities = model.predict_proba(embedding)
    
    # Label columns (same order as training)
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Calculate toxicity score
    toxicity_score = float(np.mean([prob[0][1] for prob in probabilities]))
    
    # More sensitive detection - use probability threshold instead of just predictions
    is_toxic_sensitive = toxicity_score > sensitivity or prediction.any()
    
    # Format results
    result = {
        'comment': comment,
        'predictions': {},
        'is_toxic': is_toxic_sensitive,  # More sensitive detection
        'toxicity_score': toxicity_score
    }
    
    # Add individual label predictions with sensitivity
    for i, label in enumerate(label_columns):
        prob = float(probabilities[i][0][1])
        result['predictions'][label] = {
            'prediction': bool(prediction[i]) or prob > sensitivity,  # More sensitive
            'probability': prob
        }
    
    return result

def predict_toxicity_batch(comments, sensitivity=0.3):
    """
    Batch predict toxicity for multiple comments (faster)
    
    Args:
        comments: List of comment strings
        sensitivity: Lower values = more sensitive detection
    
    Returns:
        List of prediction results
    """
    if not comments:
        return []
    
    # Get all embeddings at once
    embeddings = get_embedding(comments)
    
    # Make batch predictions
    predictions = model.predict(embeddings)
    probabilities = model.predict_proba(embeddings)
    
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    results = []
    
    for i, comment in enumerate(comments):
        prediction = predictions[i]
        probs = [prob[i] for prob in probabilities]
        
        toxicity_score = float(np.mean([prob[1] for prob in probs]))
        is_toxic_sensitive = toxicity_score > sensitivity or prediction.any()
        
        result = {
            'comment': comment,
            'predictions': {},
            'is_toxic': is_toxic_sensitive,
            'toxicity_score': toxicity_score
        }
        
        for j, label in enumerate(label_columns):
            prob = float(probs[j][1])
            result['predictions'][label] = {
                'prediction': bool(prediction[j]) or prob > sensitivity,
                'probability': prob
            }
        
        results.append(result)
    
    return results


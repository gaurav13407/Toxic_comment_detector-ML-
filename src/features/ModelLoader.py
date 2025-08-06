"""
Multi-Model Loader for Toxic Comment Detection
Supports Random Forest, LSTM, and Transformer models
"""

import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import re
from collections import Counter

# Try to import nltk, but make it optional
try:
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
models_dir = os.path.join(project_root, 'output', 'models')

class ToxicCommentMOdel(nn.Module):
    """LSTM Model class for loading saved models"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.load_available_models()
    
    def load_available_models(self):
        """Load all available models"""
        try:
            # Load Random Forest Model
            rf_path = os.path.join(models_dir, 'toxic_comment_classifier_random_forest_fast.joblib')
            if os.path.exists(rf_path):
                self.models['Random Forest'] = {
                    'model': joblib.load(rf_path),
                    'type': 'random_forest'
                }
                # Load Word2Vec for Random Forest
                w2v_path = os.path.join(models_dir, 'word2vec_model.model')
                if os.path.exists(w2v_path):
                    self.models['Random Forest']['w2v_model'] = Word2Vec.load(w2v_path)
                
                self.model_info['Random Forest'] = {
                    'description': 'Fast Random Forest with Word2Vec embeddings',
                    'accuracy': '~85%',
                    'speed': 'Very Fast'
                }
        except Exception as e:
            print(f"Could not load Random Forest model: {e}")
        
        try:
            # Load LSTM Model
            lstm_path = os.path.join(models_dir, 'lstm_toxic_classifier_full.pth')
            lstm_metadata_path = os.path.join(models_dir, 'lstm_model_metadata.pkl')
            
            if os.path.exists(lstm_path) and os.path.exists(lstm_metadata_path):
                # Load metadata
                with open(lstm_metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Load model - handle module namespace issue
                # The model was saved with __main__.ToxicCommentMOdel reference
                # We need to temporarily add our class to __main__ namespace
                import sys
                import __main__
                
                # Temporarily add our class to __main__ namespace
                __main__.ToxicCommentMOdel = ToxicCommentMOdel
                
                try:
                    lstm_model = torch.load(lstm_path, map_location='cpu', weights_only=False)
                    lstm_model.eval()
                finally:
                    # Clean up: remove from __main__ namespace
                    if hasattr(__main__, 'ToxicCommentMOdel'):
                        delattr(__main__, 'ToxicCommentMOdel')
                
                # Load the actual vocabulary and preprocessing from training data
                vocab_data = self._load_lstm_preprocessing_data()
                
                self.models['LSTM'] = {
                    'model': lstm_model,
                    'metadata': metadata,
                    'vocab_data': vocab_data,
                    'type': 'lstm'
                }
                
                self.model_info['LSTM'] = {
                    'description': 'LSTM Neural Network trained on toxic comments with NLTK tokenization',
                    'accuracy': f"~{metadata.get('hamming_accuracy', 0.85)*100:.1f}%",
                    'speed': 'Fast',
                    'details': f"Vocab: {metadata.get('vocab_size', 2000)} words, Hidden: {metadata.get('hidden_dim', 8)}"
                }
                print("✅ LSTM model loaded successfully!")
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
            import traceback
            traceback.print_exc()
        
        # Placeholder for Transformer (can be added later)
        self.model_info['Transformer (Coming Soon)'] = {
            'description': 'BERT-based transformer model (not yet implemented)',
            'accuracy': 'TBD',
            'speed': 'Slower but more accurate'
        }
    
    def _load_lstm_preprocessing_data(self):
        """Load the preprocessing data used during LSTM training"""
        try:
            # Load LSTM-specific preprocessing data
            lstm_preprocessing_path = os.path.join(models_dir, 'lstm_preprocessing_info.pkl')
            if os.path.exists(lstm_preprocessing_path):
                with open(lstm_preprocessing_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Loaded LSTM vocabulary with {data.get('vocab_size', 0)} words")
                return data
        except Exception as e:
            print(f"Warning: Could not load LSTM preprocessing data: {e}")
        
        # Fallback: Try to load general preprocessing data
        try:
            preprocessing_path = os.path.join(models_dir, 'preprocessing_info.pkl')
            if os.path.exists(preprocessing_path):
                with open(preprocessing_path, 'rb') as f:
                    return pickle.load(f)
        except:
            pass
        
        # Final fallback: Create a basic vocabulary for LSTM prediction
        print("Warning: Using fallback vocabulary for LSTM")
        return {
            'word2indx': {'<PAD>': 0, '<UNK>': 1},
            'max_length': 100,
            'max_vocab_size': 2000
        }
    
    def _create_simple_vocab(self):
        """Create a simple vocabulary for LSTM model (deprecated - use _load_lstm_preprocessing_data)"""
        return {'<PAD>': 0, '<UNK>': 1}
    
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())
    
    def get_model_info(self):
        """Return information about all models"""
        return self.model_info
    
    def predict_toxicity(self, comment, model_name='Random Forest', sensitivity=0.3):
        """
        Predict toxicity using specified model
        
        Args:
            comment: Text to analyze
            model_name: Name of model to use
            sensitivity: Detection sensitivity
        
        Returns:
            dict: Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        model_data = self.models[model_name]
        model_type = model_data['type']
        
        if model_type == 'random_forest':
            return self._predict_random_forest(comment, model_data, sensitivity)
        elif model_type == 'lstm':
            return self._predict_lstm(comment, model_data, sensitivity)
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
    
    def _predict_random_forest(self, comment, model_data, sensitivity):
        """Predict using Random Forest model"""
        model = model_data['model']
        w2v_model = model_data.get('w2v_model')
        
        if not w2v_model:
            raise ValueError("Word2Vec model not loaded for Random Forest")
        
        # Preprocess and get embedding
        embedding = self._get_rf_embedding(comment, w2v_model)
        
        # Make prediction
        prediction = model.predict(embedding)[0]
        probabilities = model.predict_proba(embedding)
        
        # Label columns
        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Calculate toxicity score
        toxicity_score = float(np.mean([prob[0][1] for prob in probabilities]))
        
        # Sensitive detection
        is_toxic_sensitive = toxicity_score > sensitivity or prediction.any()
        
        # Format results
        result = {
            'comment': comment,
            'predictions': {},
            'is_toxic': is_toxic_sensitive,
            'toxicity_score': toxicity_score,
            'model_used': 'Random Forest'
        }
        
        for i, label in enumerate(label_columns):
            prob = float(probabilities[i][0][1])
            result['predictions'][label] = {
                'prediction': bool(prediction[i]) or prob > sensitivity,
                'probability': prob
            }
        
        return result
    
    def _predict_lstm(self, comment, model_data, sensitivity, threshold=0.5):
        """Predict using LSTM model with proper tokenization"""
        model = model_data['model']
        metadata = model_data['metadata']
        vocab_data = model_data.get('vocab_data', {})
        
        # Get preprocessing parameters
        max_length = vocab_data.get('max_length', 100)
        max_vocab_size = metadata.get('max_vocab_size', 2000)
        word2indx = vocab_data.get('word2indx', {'<PAD>': 0, '<UNK>': 1})
        
        # Tokenize text using NLTK if available, otherwise use simple split
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(comment.lower())
            except:
                tokens = comment.lower().split()
        else:
            tokens = comment.lower().split()
        
        # Encode tokens to indices
        def encode_tokens(tokens, word2indx, max_vocab):
            indices = []
            for token in tokens[:max_length]:
                idx = word2indx.get(token, 1)  # Use UNK token if not found
                if idx >= max_vocab:
                    idx = 1  # Replace with UNK if out of range
                indices.append(idx)
            return indices
        
        # If we have a proper word2indx vocabulary, use it
        if len(word2indx) > 2:  # More than just PAD and UNK
            indices = encode_tokens(tokens, word2indx, max_vocab_size)
        else:
            # Fallback: create simple hash-based encoding
            indices = []
            for token in tokens[:max_length]:
                # Simple hash to index mapping
                idx = hash(token) % (max_vocab_size - 2) + 2
                indices.append(idx)
        
        # Pad sequence
        while len(indices) < max_length:
            indices.append(0)
        
        # Convert to tensor
        text_tensor = torch.tensor([indices], dtype=torch.long)
        
        # Predict
        with torch.no_grad():
            output = model(text_tensor)
            probabilities = torch.sigmoid(output).squeeze().numpy()
            predictions = (probabilities > threshold).astype(int)
        
        # Calculate overall toxicity score
        toxicity_score = float(np.max(probabilities))
        is_toxic = any(predictions) or toxicity_score > sensitivity
        
        # Format results
        label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        result = {
            'comment': comment,
            'predictions': {},
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'model_used': 'LSTM'
        }
        
        for i, label in enumerate(label_names):
            result['predictions'][label] = {
                'prediction': bool(predictions[i]),
                'probability': float(probabilities[i])
            }
        
        return result
    
    def _get_rf_embedding(self, text, w2v_model, vector_size=300):
        """Get Word2Vec embedding for Random Forest model"""
        if not text or pd.isna(text):
            return np.zeros((1, vector_size))
        
        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = simple_preprocess(text, deacc=True, min_len=2, max_len=15)
        
        if not tokens:
            return np.zeros((1, vector_size))
        
        # Get word vectors
        word_vectors = []
        for token in tokens:
            if token in w2v_model.wv:
                word_vectors.append(w2v_model.wv[token])
        
        if not word_vectors:
            return np.zeros((1, vector_size))
        
        # Average the word vectors
        embedding = np.mean(word_vectors, axis=0)
        return embedding.reshape(1, -1)

# Global model loader instance
model_loader = ModelLoader()

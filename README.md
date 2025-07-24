# Toxic Comment Detector ML

A machine learning system for detecting toxic comments across multiple platforms (YouTube, Reddit, Twitter) using Word2Vec embeddings and Random Forest classification.

## Features

- **Multi-platform support**: Analyze comments from YouTube, Reddit, and Twitter
- **High-speed processing**: Word2Vec embeddings with 2,025 texts/sec processing speed
- **Accurate detection**: Random Forest classifier achieving 89.73% accuracy
- **Web interface**: User-friendly Streamlit app with adjustable sensitivity
- **Batch processing**: Efficient analysis of large comment datasets
- **Flexible limits**: Configurable comment fetching (100, 500, 1000, 2000, or unlimited)

## Project Structure

```
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── Prediction.ipynb           # Model training notebook
├── src/
│   ├── features/
│   │   └── Detection.py           # Core prediction pipeline
│   └── preprocessing/
│       ├── yt_scraper.py          # YouTube comment scraper
│       ├── Reddit_scraper.py      # Reddit comment scraper
│       └── twitter.py             # Twitter scraper (disabled)
├── data/                          # Training data directory
├── output/
│   └── models/                    # Trained models (excluded from repo)
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/gaurav13407/Toxic_comment_detector-ML-.git
cd Toxic_comment_detector-ML-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
The trained models are not included in the repository due to size constraints. You need to train them yourself:

1. Open and run `notebooks/Prediction.ipynb` to train the Word2Vec and Random Forest models
2. This will generate the required model files in `output/models/`:
   - `word2vec_model.model` (Word2Vec embeddings)
   - `toxic_comment_classifier_random_forest_medium.joblib` (Random Forest classifier)
   - Supporting files for preprocessing

### 4. Configure API credentials (for Reddit)
Create a `.env` file in the root directory:
```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_app_name
```

## Usage

### Web Interface
Run the Streamlit app:
```bash
streamlit run app.py
```

The web interface provides:
- **Platform selection**: Choose YouTube, Reddit, or Twitter
- **Sensitivity control**: Adjust detection threshold (0.1-0.5)
- **Comment limits**: Set how many comments to fetch
- **Real-time analysis**: View toxicity predictions with confidence scores

### Python API
```python
from src.features.Detection import predict_toxicity, predict_toxicity_batch

# Single comment analysis
result = predict_toxicity("This is a comment", sensitivity=0.3)
print(result['is_toxic'])

# Batch analysis (faster for multiple comments)
comments = ["Comment 1", "Comment 2", "Comment 3"]
results = predict_toxicity_batch(comments, sensitivity=0.3)
```

## Model Performance

- **Processing Speed**: 2,025 texts/second with Word2Vec embeddings
- **Accuracy**: 89.73% on test dataset
- **Labels**: Detects 6 types of toxicity:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate

## Technical Details

### Word2Vec Configuration
- **Vector Size**: 300 dimensions
- **Vocabulary**: 93,433 unique words
- **Training**: Optimized for toxic comment detection

### Random Forest Configuration
- **Multi-output classification**: Predicts all 6 toxicity labels simultaneously
- **Ensemble method**: Robust performance across different comment types
- **Sensitivity tuning**: Adjustable threshold for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Dependencies

Key libraries:
- `streamlit`: Web interface
- `scikit-learn`: Machine learning models
- `gensim`: Word2Vec embeddings
- `pandas`, `numpy`: Data processing
- `youtube-comment-downloader`: YouTube API
- `praw`: Reddit API
- `snscrape`: Twitter scraping (disabled)

## License

This project is open source. Please ensure compliance with platform APIs when scraping comments.

## Troubleshooting

### Common Issues

1. **Missing model files**: Run the training notebook first
2. **API rate limits**: Reduce comment limits or add delays
3. **Sensitivity too low**: Increase sensitivity value (0.3-0.5)
4. **Twitter scraper disabled**: snscrape compatibility issues

### Performance Tips

- Use batch processing for analyzing many comments
- Adjust sensitivity based on your use case
- Monitor API rate limits when scraping
- Consider using larger comment limits for better context

## Acknowledgments

Built with modern ML best practices and optimized for real-world toxic comment detection scenarios.

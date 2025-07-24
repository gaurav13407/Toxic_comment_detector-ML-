import streamlit as st
import pandas as pd
from src.preprocessing.Reddit_scraper import fetch_reddit_comments
from src.preprocessing.twitter import fetch_twitter_comments
from src.preprocessing.yt_scraper import fetch_youtube_comments
from src.features.Detection import predict_toxicity_batch

st.title("💬 Multi-Platform Toxic Comment Detector")
st.sidebar.markdown("### Settings")
sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 0.8, 0.25, 0.05, 
                                help="Lower = more sensitive (catches more toxic comments)")

# Comment limit setting
comment_limit = st.sidebar.selectbox(
    "Comment Limit", 
    [100, 500, 1000, 2000, "All"], 
    index=3,  # Default to 2000
    help="Maximum number of comments to fetch and analyze"
)

# Convert "All" to None for unlimited
if comment_limit == "All":
    comment_limit = None

platform = st.selectbox("Select Platform", ["YouTube", "Twitter", "Reddit"])
url = st.text_input("Enter the post/video URL")

if url:
    with st.spinner(f"Fetching {platform} comments (limit: {comment_limit or 'unlimited'})..."):
        try:
            if platform == "YouTube":
                comments = fetch_youtube_comments(url, limit=comment_limit)
            elif platform == "Twitter":
                comments = fetch_twitter_comments(url)
            elif platform == "Reddit":
                comments = fetch_reddit_comments(url, limit=comment_limit)
            
            if comments:
                st.success(f"Found {len(comments)} comments!")
                
                users, texts = zip(*comments)
                
                # Get toxicity predictions using BATCH processing (faster)
                with st.spinner("Analyzing comments for toxicity..."):
                    results = predict_toxicity_batch(list(texts), sensitivity=sensitivity)
                
                # Create DataFrame
                df_data = []
                for i, result in enumerate(results):
                    toxic_labels = [label for label, pred in result['predictions'].items() 
                                  if pred['prediction']]
                    
                    df_data.append({
                        "Username": users[i],
                        "Comment": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                        "Toxic": "🚨 YES" if result['is_toxic'] else "✅ NO",
                        "Score": f"{result['toxicity_score']:.3f}",
                        "Labels": ", ".join(toxic_labels) if toxic_labels else "Clean"
                    })
                
                df = pd.DataFrame(df_data)
                
                # Show statistics
                toxic_count = sum(1 for r in results if r['is_toxic'])
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Comments", len(comments))
                with col2:
                    st.metric("Toxic Comments", toxic_count, 
                             f"{100*toxic_count/len(comments):.1f}%")
                with col3:
                    st.metric("Clean Comments", len(comments) - toxic_count)
                
                # Filter options
                show_filter = st.radio("Show:", ["All Comments", "Toxic Only", "Clean Only"])
                
                if show_filter == "Toxic Only":
                    df = df[df['Toxic'] == "🚨 YES"]
                elif show_filter == "Clean Only":
                    df = df[df['Toxic'] == "✅ NO"]
                
                # Display results
                st.dataframe(df, use_container_width=True)
                
                if toxic_count > 0:
                    st.warning(f"⚠️ Found {toxic_count} potentially toxic comments. Review manually for accuracy.")
                
            else:
                st.warning("No comments found. Please check the URL.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please make sure the URL is valid and accessible.")

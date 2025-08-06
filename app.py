import streamlit as st
import pandas as pd
from src.preprocessing.Reddit_scraper import fetch_reddit_comments
from src.preprocessing.twitter import fetch_twitter_comments
from src.preprocessing.yt_scraper import fetch_youtube_comments
from src.features.Detection import (
    predict_toxicity_batch, 
    get_available_models, 
    get_model_info,
    predict_toxicity_batch_with_model
)

st.set_page_config(
    page_title="Multi-Platform Toxic Comment Detector",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Multi-Platform Toxic Comment Detector")
st.markdown("---")

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Model Settings")

# Model Selection
available_models = get_available_models()
model_info = get_model_info()

if available_models:
    selected_model = st.sidebar.selectbox(
        "🤖 Select Model",
        available_models,
        help="Choose the AI model for toxicity detection"
    )
    
    # Show model information
    if selected_model in model_info:
        info = model_info[selected_model]
        st.sidebar.markdown(f"**📊 Model Info:**")
        st.sidebar.markdown(f"- **Description:** {info['description']}")
        st.sidebar.markdown(f"- **Accuracy:** {info['accuracy']}")
        st.sidebar.markdown(f"- **Speed:** {info['speed']}")
else:
    st.sidebar.error("❌ No models available!")
    selected_model = "Random Forest"  # Fallback

st.sidebar.markdown("---")

# Detection Settings
sensitivity = st.sidebar.slider(
    "🎯 Detection Sensitivity", 
    0.1, 0.8, 0.25, 0.05,
    help="Lower = more sensitive (catches more toxic comments)"
)

# Comment limit setting
comment_limit = st.sidebar.selectbox(
    "📊 Comment Limit", 
    [100, 500, 1000, 2000, "All"], 
    index=2,  # Default to 1000
    help="Maximum number of comments to fetch and analyze"
)

# Convert "All" to None for unlimited
if comment_limit == "All":
    comment_limit = None

st.sidebar.markdown("---")

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    platform = st.selectbox("🌐 Select Platform", ["YouTube", "Twitter", "Reddit"])
    url = st.text_input("🔗 Enter the post/video URL", placeholder="Paste URL here...")

with col2:
    st.markdown("### 📈 Quick Stats")
    if 'results' in st.session_state:
        results = st.session_state.results
        toxic_count = sum(1 for r in results if r['is_toxic'])
        total_count = len(results)
        
        st.metric("Total", total_count)
        st.metric("Toxic", toxic_count, f"{100*toxic_count/total_count:.1f}%" if total_count > 0 else "0%")
        st.metric("Model", st.session_state.get('model_used', selected_model))

if url:
    with st.spinner(f"🔍 Fetching {platform} comments (limit: {comment_limit or 'unlimited'})..."):
        try:
            if platform == "YouTube":
                comments = fetch_youtube_comments(url, limit=comment_limit)
            elif platform == "Twitter":
                comments = fetch_twitter_comments(url)
            elif platform == "Reddit":
                comments = fetch_reddit_comments(url, limit=comment_limit)
            
            if comments:
                st.success(f"✅ Found {len(comments)} comments!")
                
                users, texts = zip(*comments)
                
                # Get toxicity predictions using selected model
                with st.spinner(f"🧠 Analyzing comments with {selected_model} model..."):
                    if selected_model in available_models:
                        results = predict_toxicity_batch_with_model(
                            list(texts), 
                            model_name=selected_model, 
                            sensitivity=sensitivity
                        )
                    else:
                        # Fallback to default model
                        results = predict_toxicity_batch(list(texts), sensitivity=sensitivity)
                        for result in results:
                            result['model_used'] = 'Random Forest (Default)'
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.model_used = selected_model
                
                # Create DataFrame with enhanced information
                df_data = []
                for i, result in enumerate(results):
                    # Get toxic labels
                    toxic_labels = []
                    for label, pred in result['predictions'].items():
                        if isinstance(pred, dict) and pred.get('prediction'):
                            toxic_labels.append(label)
                        elif pred and not isinstance(pred, dict):  # Handle old format
                            toxic_labels.append(label)
                    
                    # Format comment preview
                    comment_preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
                    
                    df_data.append({
                        "👤 Username": users[i],
                        "💬 Comment": comment_preview,
                        "🚨 Toxic": "🚨 YES" if result['is_toxic'] else "✅ NO",
                        "📊 Score": f"{result['toxicity_score']:.3f}",
                        "🏷️ Labels": ", ".join(toxic_labels) if toxic_labels else "Clean",
                        "🤖 Model": result.get('model_used', selected_model)
                    })
                
                df = pd.DataFrame(df_data)
                
                # Enhanced Statistics
                toxic_count = sum(1 for r in results if r['is_toxic'])
                st.markdown("### 📊 Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Total Comments", len(comments))
                with col2:
                    st.metric("🚨 Toxic Comments", toxic_count, 
                             f"{100*toxic_count/len(comments):.1f}%")
                with col3:
                    st.metric("✅ Clean Comments", len(comments) - toxic_count)
                with col4:
                    st.metric("🤖 Model Used", selected_model)
                
                # Category breakdown
                if toxic_count > 0:
                    st.markdown("### 🏷️ Toxicity Categories")
                    category_counts = {}
                    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                    
                    for label in label_names:
                        count = 0
                        for result in results:
                            pred = result['predictions'].get(label, {})
                            if isinstance(pred, dict) and pred.get('prediction'):
                                count += 1
                            elif pred and not isinstance(pred, dict):
                                count += 1
                        category_counts[label] = count
                    
                    # Display in columns
                    cols = st.columns(3)
                    for i, (label, count) in enumerate(category_counts.items()):
                        with cols[i % 3]:
                            percentage = (count / len(comments)) * 100 if len(comments) > 0 else 0
                            st.metric(
                                f"{label.replace('_', ' ').title()}", 
                                count, 
                                f"{percentage:.1f}%"
                            )
                
                # Filter options
                st.markdown("### 🔍 Filter Results")
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    show_filter = st.radio(
                        "Show Comments:", 
                        ["All Comments", "Toxic Only", "Clean Only"],
                        horizontal=True
                    )
                
                with filter_col2:
                    if toxic_count > 0:
                        category_filter = st.selectbox(
                            "Filter by Category:",
                            ["All Categories"] + [label.replace('_', ' ').title() for label in label_names]
                        )
                    else:
                        category_filter = "All Categories"
                
                # Apply filters
                filtered_df = df.copy()
                
                if show_filter == "Toxic Only":
                    filtered_df = filtered_df[filtered_df['🚨 Toxic'] == "🚨 YES"]
                elif show_filter == "Clean Only":
                    filtered_df = filtered_df[filtered_df['🚨 Toxic'] == "✅ NO"]
                
                if category_filter != "All Categories" and toxic_count > 0:
                    category_lower = category_filter.lower().replace(' ', '_')
                    filtered_df = filtered_df[filtered_df['🏷️ Labels'].str.contains(category_lower, case=False, na=False)]
                
                # Display results
                st.markdown("### 📋 Detailed Results")
                st.dataframe(filtered_df, use_container_width=True, height=400)
                
                # Download option
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"toxic_comments_{platform.lower()}_{selected_model.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
                
                # Warnings and insights
                if toxic_count > 0:
                    st.warning(f"⚠️ Found {toxic_count} potentially toxic comments using {selected_model} model. Review manually for accuracy.")
                    
                    # Model-specific insights
                    if selected_model == "LSTM":
                        st.info("💡 **LSTM Model**: Good at understanding context and sequence patterns in text.")
                    elif selected_model == "Random Forest":
                        st.info("💡 **Random Forest Model**: Fast and reliable for general toxicity detection.")
                else:
                    st.success("🎉 Great! No toxic comments detected. This appears to be a healthy discussion.")
                
            else:
                st.warning("⚠️ No comments found. Please check the URL and try again.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Please make sure the URL is valid and accessible. Some platforms may have restrictions.")

else:
    # Show example URLs when no URL is entered
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Select a Model** from the sidebar (Random Forest for speed, LSTM for accuracy)
    2. **Choose Platform** (YouTube, Twitter, or Reddit)
    3. **Paste URL** of the post/video you want to analyze
    4. **Adjust Sensitivity** if needed (lower = more sensitive detection)
    5. **Click Analyze** and review the results!
    """)
    
    st.markdown("### 📝 Example URLs")
    st.markdown("""
    - **YouTube**: `https://www.youtube.com/watch?v=VIDEO_ID`
    - **Twitter**: `https://twitter.com/user/status/TWEET_ID`
    - **Reddit**: `https://www.reddit.com/r/subreddit/comments/POST_ID/`
    """)
    
    # Model comparison
    if model_info:
        st.markdown("### 🤖 Available Models")
        model_df = pd.DataFrame([
            {
                "Model": model_name,
                "Description": info["description"],
                "Accuracy": info["accuracy"],
                "Speed": info["speed"]
            }
            for model_name, info in model_info.items()
        ])
        st.dataframe(model_df, use_container_width=True)

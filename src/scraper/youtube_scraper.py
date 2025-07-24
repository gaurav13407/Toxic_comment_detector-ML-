from googleapiclient.discovery import build
import pandas as pd
import os
import sys
from dotenv import load_dotenv
# from src.utils import logger  # Commented out for now
import pandas as pd

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

load_dotenv()

API_KEY = os.getenv('YOUTUBE_API_KEY')

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_youtube_comments(video_id, max_comments=1000000):
    comments=[]
    next_page_token=None

    while(len(comments) < max_comments):
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Fixed parameter name and set reasonable limit
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response['items']:
            comment=item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if(len(comments)>=max_comments):
                break

        next_page_token=response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

#Example 

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2]
        elif query.path.startswith('/shorts/'):  # Added support for YouTube Shorts
            return query.path.split('/')[2]
    return None

# Example:
url = input("Enter YouTube video URL: ")
video_id = extract_video_id(url)

print(f"Extracted video ID: {video_id}")

if video_id:
    print(f"Fetching comments for video ID: {video_id}")
    comments = get_youtube_comments(video_id, max_comments=50000)  # Reduced for testing
    
    print(f"Found {len(comments)} comments")
    
    # NEW: Save comments to hindi_train.csv in data/raw folder
    os.makedirs('data/raw', exist_ok=True)
    df = pd.DataFrame({'comment': comments})
    output_file = 'data/raw/hindi_train.csv'
    
    # Check if file exists to append or create new
    if os.path.exists(output_file):
        # Append to existing file
        df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
        print(f"Added {len(comments)} comments to existing {output_file}")
    else:
        # Create new file with header
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Created new file {output_file} with {len(comments)} comments")
    

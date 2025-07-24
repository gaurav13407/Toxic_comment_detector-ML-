import instaloader
import csv
import os
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create an instance of Instaloader with more human-like settings
L = instaloader.Instaloader(
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    sleep=True,  # Enable automatic sleep between requests
    quiet=False  # Show what's happening
)

print("Initializing Instagram connection...")
time.sleep(random.uniform(2, 5))  # Random delay to appear more human

# Login to Instagram (get credentials from environment variables)
USERNAME = os.getenv('INSTAGRAM_USERNAME')
PASSWORD = os.getenv('INSTAGRAM_PASSWORD')

if not USERNAME or not PASSWORD:
    print("Please set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in your .env file")
    exit(1)

try:
    print("Attempting to login to Instagram...")
    print("Note: If you see 'change password' page, please:")
    print("1. Login manually in browser first")
    print("2. Complete any security challenges")
    print("3. Wait 15-30 minutes, then try again")
    
    L.login(USERNAME, PASSWORD)
    print("Successfully logged into Instagram!")
    time.sleep(random.uniform(3, 7))  # Wait after login
except Exception as e:
    print(f"Login failed: {e}")
    print("\n🔧 TROUBLESHOOTING TIPS:")
    print("1. Open Instagram in browser and login manually first")
    print("2. Check if your account has 2FA enabled - disable temporarily")
    print("3. Use an app-specific password if available")
    print("4. Try logging in from the same device/network you normally use")
    print("5. Wait 24 hours if you've had multiple failed attempts")
    exit(1)

# Get Instagram handle from user input
instagram_handle = input("Enter Instagram handle (without @): ")

# Load the profile
try:
    profile = instaloader.Profile.from_username(L.context, instagram_handle)
    print(f"Found profile: {profile.username} with {profile.mediacount} posts")
except Exception as e:
    print(f"Failed to load profile {instagram_handle}: {e}")
    exit(1)

# Prepare data container
data = []
comments_data = []  # Separate list for individual comments

print("Scraping posts and comments...")

# Loop over posts (limit to recent posts for faster processing)
post_count = 0
max_posts = 20  # Limit for faster processing

for post in profile.get_posts():
    if post_count >= max_posts:
        break
    
    try:
        print(f"Processing post {post_count + 1}/{max_posts}: {post.shortcode}")
        comments = []
        
        # Get comments from this post
        for comment in post.get_comments():
            comment_text = comment.text
            comments.append(comment_text)
            
            # Also save individual comments for ML training
            comments_data.append({
                'comment': comment_text,
                'post_id': post.shortcode,
                'username': comment.owner.username
            })

        post_data = {
            'PostID': post.shortcode,
            'Likes': post.likes,
            'CommentsCount': post.comments,
            'Comments': " ||| ".join(comments[:10])  # first 10 comments joined
        }
        data.append(post_data)
        post_count += 1
        
    except Exception as e:
        print(f"Error processing post {post.shortcode}: {e}")
        continue

# Create data/raw directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save post data to CSV
csv_file = f"data/raw/{instagram_handle}_posts_data.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['PostID', 'Likes', 'CommentsCount', 'Comments'])
    writer.writeheader()
    for entry in data:
        writer.writerow(entry)

# Save individual comments to insta_comm.csv
comments_file = 'data/raw/insta_comm.csv'
if comments_data:
    import pandas as pd
    df = pd.DataFrame(comments_data)
    
    # Check if file exists to append or create new
    if os.path.exists(comments_file):
        # Append only the comment text to existing file
        comment_df = pd.DataFrame({'comment': [item['comment'] for item in comments_data]})
        comment_df.to_csv(comments_file, mode='a', header=False, index=False, encoding='utf-8')
        print(f"Added {len(comments_data)} comments to existing {comments_file}")
    else:
        # Create new file with header
        comment_df = pd.DataFrame({'comment': [item['comment'] for item in comments_data]})
        comment_df.to_csv(comments_file, index=False, encoding='utf-8')
        print(f"Created new file {comments_file} with {len(comments_data)} comments")

print(f"Scraped data for {instagram_handle}:")
print(f"- Posts data saved to: {csv_file}")
print(f"- Comments saved to: {comments_file}")
print(f"- Total posts processed: {len(data)}")
print(f"- Total comments collected: {len(comments_data)}")

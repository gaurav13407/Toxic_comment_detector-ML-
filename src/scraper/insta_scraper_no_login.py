import instaloader
import csv
import os
import pandas as pd

# Create an instance of Instaloader (no login required for public profiles)
L = instaloader.Instaloader()

print("=== Instagram Scraper (No Login Required) ===")
print("Note: This works only for public profiles and may have limited access to comments")

# Get Instagram handle from user input
instagram_handle = input("Enter Instagram handle (without @): ")

# Load the profile
try:
    profile = instaloader.Profile.from_username(L.context, instagram_handle)
    print(f"Found profile: {profile.username} with {profile.mediacount} posts")
    
    if profile.is_private:
        print("❌ This profile is private. You need to login and follow this account to access content.")
        exit(1)
    else:
        print("✅ This is a public profile. Proceeding with scraping...")
        
except Exception as e:
    print(f"Failed to load profile {instagram_handle}: {e}")
    exit(1)

# Prepare data container
data = []
comments_data = []  # Separate list for individual comments

print("Scraping posts and comments...")

# Loop over posts (limit to recent posts for faster processing)
post_count = 0
max_posts = 10  # Reduced limit for public access

for post in profile.get_posts():
    if post_count >= max_posts:
        break
    
    try:
        print(f"Processing post {post_count + 1}/{max_posts}: {post.shortcode}")
        comments = []
        comment_count = 0
        
        # Get comments from this post (may be limited for public access)
        try:
            for comment in post.get_comments():
                comment_text = comment.text
                comments.append(comment_text)
                
                # Also save individual comments for ML training
                comments_data.append({
                    'comment': comment_text,
                    'post_id': post.shortcode,
                    'username': comment.owner.username
                })
                comment_count += 1
                
                # Limit comments per post to avoid rate limiting
                if comment_count >= 50:
                    break
                    
        except Exception as comment_error:
            print(f"  ⚠️ Could not access comments for this post: {comment_error}")
            # Continue with post data even if comments fail

        post_data = {
            'PostID': post.shortcode,
            'Likes': post.likes,
            'CommentsCount': post.comments,
            'Comments': " ||| ".join(comments[:10])  # first 10 comments joined
        }
        data.append(post_data)
        post_count += 1
        
        print(f"  ✅ Collected {len(comments)} comments from this post")
        
    except Exception as e:
        print(f"  ❌ Error processing post: {e}")
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

print(f"\n🎉 Scraping completed for {instagram_handle}:")
print(f"- Posts data saved to: {csv_file}")
print(f"- Comments saved to: {comments_file}")
print(f"- Total posts processed: {len(data)}")
print(f"- Total comments collected: {len(comments_data)}")

if len(comments_data) == 0:
    print("\n⚠️ No comments were collected. This could be because:")
    print("  1. The posts have no comments")
    print("  2. Instagram is limiting access to comments")
    print("  3. The profile requires login to view comments")
    print("\nTry using your YouTube scraper instead, which works reliably!")

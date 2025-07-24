from youtube_comment_downloader import YoutubeCommentDownloader

def fetch_youtube_comments(video_url, limit=None):
    """
    Fetch YouTube comments with option to get all comments
    
    Args:
        video_url: YouTube video URL
        limit: Maximum number of comments (None = all comments)
    """
    try:
        downloader = YoutubeCommentDownloader()
        
        # Get comments generator
        comments_generator = downloader.get_comments_from_url(video_url, sort_by=1)  # sort_by=1 for top comments
        
        comments = []
        count = 0
        
        for comment in comments_generator:
            comments.append((comment['author'], comment['text']))
            count += 1
            
            # Break if we reach the limit
            if limit and count >= limit:
                break
                
            # Optional: Add a reasonable upper limit to prevent infinite loading
            if count >= 5000:  # Max 5000 comments to prevent memory issues
                break
        
        return comments
        
    except Exception as e:
        print(f"Error fetching YouTube comments: {e}")
        return []

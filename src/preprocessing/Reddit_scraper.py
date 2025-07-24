import praw

reddit = praw.Reddit(
    client_id="G8CEXPqn83IebQbnkYrZ5Q",
    client_secret="BMu9I1ZLXuuoIcHOvoJW5X_HZRm93w",
    user_agent="SentimentScraper by u/Gaurav_127127"
)


def fetch_reddit_comments(post_url, limit=None):
    """
    Fetch Reddit comments with optional limit
    
    Args:
        post_url: Reddit post URL
        limit: Maximum number of comments (None = all comments)
    """
    try:
        submission = reddit.submission(url=post_url)
        submission.comments.replace_more(limit=0)  # Don't expand "more comments" to speed up
        
        comments = []
        count = 0
        
        for comment in submission.comments.list():
            if hasattr(comment, 'author') and hasattr(comment, 'body'):
                author = comment.author.name if comment.author else "Unknown"
                comments.append((author, comment.body))
                count += 1
                
                # Break if we reach the limit
                if limit and count >= limit:
                    break
        
        return comments
        
    except Exception as e:
        print(f"Error fetching Reddit comments: {e}")
        return []
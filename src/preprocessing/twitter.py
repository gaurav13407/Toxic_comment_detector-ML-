def fetch_twitter_comments(tweet_url):
    """
    Fetch Twitter comments/replies - currently disabled due to API restrictions
    """
    # Twitter scraping is currently not working due to API changes and restrictions
    # Return empty list for now
    print("⚠️ Twitter scraping is currently disabled due to API restrictions")
    return []

# Alternative implementation (commented out due to snscrape issues)
"""
import snscrape.modules.twitter as sntwitter

def fetch_twitter_comments(tweet_url):
    tweet_id = tweet_url.split("/")[-1]
    replies = []
    for tweet in sntwitter.TwitterTweetScraper(tweet_id, mode='replies').get_items():
        replies.append((tweet.user.username, tweet.content))
    return replies
"""

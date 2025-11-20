import tweepy
import pandas as pd

# consumer_key = "o9XEVYuYkJ6jumEcmpCKU9CPq"
# consumer_secret = "1aRuwtXo7BbkY1f9qj6l4j5rRxB5uNXE593EzZ8ENzhHvhMbnl"
# bearer_token1="AAAAAAAAAAAAAAAAAAAAANkp4wEAAAAAYN2CWGEgOEu0NTpbzdNdSxQB6%2BA%3DUzQmqMJcTMYSa0DOLI7hST7qF71Sl07Zle5Fj7i7btTgV5y1A6"

# bearer_token_iphone="AAAAAAAAAAAAAAAAAAAAADbh4wEAAAAAxh5Q%2BLro%2BAvU1oRjWAJU4zt%2BYxY%3DNBWXuO8vtqnQ76Weo2ia0TQx2yfVBVCRit0XEd0q9j5tMGRHDy"
bearer_token="AAAAAAAAAAAAAAAAAAAAAKHh4wEAAAAAvjr5jjeGXHroCp3XhF1i0aJUEBc%3D0A3IBlh0DxcNj8PeXKRjz1mmpRs68vzMxeBKqOxOPxSrqX12dg"
# auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
client = tweepy.Client(bearer_token, wait_on_rate_limit=True)


# ----------------------------
# 🔍 QUERY SETTINGS
# ----------------------------
# Query: look for English tweets mentioning Duolingo
# "-is:retweet" ensures we skip retweets (as per data usage policy)
query = "(Duolingo OR @duolingo OR #Duolingo) -is:retweet lang:en"

# Maximum number of tweets allowed by free-tier API (v2)
max_results = 100

# ----------------------------
# 📥 FETCH DATA
# ----------------------------
tweets = client.search_recent_tweets(
    query=query,
    max_results=max_results,
    tweet_fields=["id", "text", "created_at", "public_metrics", "lang"]
)

if not tweets.data:
    print("No tweets found.")
    exit()

# ----------------------------
# 🧱 STRUCTURE DATA
# ----------------------------
data = []
for tweet in tweets.data:
    metrics = tweet.public_metrics
    data.append({
        "tweet_id": tweet.id,
        "created_at": tweet.created_at,
        "text": tweet.text,
        "likes": metrics.get("like_count", 0),
        "retweets": metrics.get("retweet_count", 0),
        "replies": metrics.get("reply_count", 0),
        "quotes": metrics.get("quote_count", 0)
    })

# Create DataFrame
df = pd.DataFrame(data)

# ----------------------------
# 💾 SAVE DATA
# ----------------------------
output_file = f"duolingo_tweets.csv"

# Save text + public metrics only (compliant with Twitter policy)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ Saved {len(df)} tweets to {output_file}")
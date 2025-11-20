import asyncio
import csv
from twikit import Client, TooManyRequests
from random import randint

# QUERY = '(from:elonmusk) lang:en until:2020-01-01 since:2018-01-01'
query = '(("iPhone 17" OR #iPhone17 OR "iPhone 17 Pro" OR "Apple Event") \
 -giveaway -win -free -deal -discount -referral -offer -promo -shop -case -cover \
 -#ad -#sponsored -filter:retweets) lang:en until:2025-10-30 since:2025-09-09'

USERNAME = 'KaterynaStudent'
EMAIL = 'kateryna.severyna.student@gmail.com'
PASSWORD = 'Slonik14k'

# Initialize client
client = Client('en-US')
OUTPUT_FILE = "tweets_iphone17.csv"

async def main():
    client.load_cookies('cookies.json')

    all_tweets = []
    cursor = None

    # Open CSV file and prepare headers
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text", "retweets", "likes", "replies", "quotes"])

        while len(all_tweets) < 10000:
            try:
                result = await client.search_tweet(
                    query=query,
                    product='Latest',
                    count=20,
                    cursor=cursor
                )

                tweets = list(result)
                all_tweets.extend(tweets)

                # ✅ Write tweets to CSV as we go
                for t in tweets:
                    writer.writerow([
                        t.id,
                        t.created_at,
                        t.full_text.replace("\n", " "),  # remove line breaks
                        t.retweet_count,
                        t.favorite_count,
                        t.reply_count,
                        t.quote_count
                    ])

                print(f"Fetched {len(all_tweets)} tweets so far...")

                # Stop if no next page
                if not result.next_cursor:
                    break

                cursor = result.next_cursor
                await asyncio.sleep(randint(3, 7))  # variable delay to avoid rate-limits

            except TooManyRequests:
                wait_time = 900  # 15 minutes
                print(f"Rate limit reached. Waiting {wait_time/60} minutes...")
                await asyncio.sleep(wait_time)
                continue  # retry same cursor

    print(f"✅ Done. Saved {len(all_tweets)} tweets to '{OUTPUT_FILE}'.")


asyncio.run(main())

# tweet_count = 0
# tweets = None
#
# while tweet_count < MINIMUM_TWEETS:
#
#     try:
#         tweets = get_tweets(tweets)
#     except TooManyRequests as e:
#         rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
#         print(f'{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}')
#         wait_time = rate_limit_reset - datetime.now()
#         time.sleep(wait_time.total_seconds())
#         continue
#
#     if not tweets:
#         print(f'{datetime.now()} - No more tweets found')
#         break
#
#     for tweet in tweets:
#         tweet_count += 1
#         tweet_data = [tweet_count, tweet.user.name, tweet.text, tweet.created_at, tweet.retweet_count,
#                       tweet.favorite_count]
#
#         with open('tweets.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(tweet_data)
#
#     print(f'{datetime.now()} - Got {tweet_count} tweets')
#
# print(f'{datetime.now()} - Done! Got {tweet_count} tweets found')
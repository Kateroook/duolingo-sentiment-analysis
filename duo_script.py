import asyncio
import csv
from twikit import Client, TooManyRequests
from random import randint

# QUERY = '(from:elonmusk) lang:en until:2020-01-01 since:2018-01-01'
query = '(Duolingo OR @duolingo OR #Duolingo) -is:retweet lang:en'

# Initialize client
client = Client('en-US')
OUTPUT_FILE = "tweets_duolingo.csv"

async def main():
    client.load_cookies('duocookies.json')

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
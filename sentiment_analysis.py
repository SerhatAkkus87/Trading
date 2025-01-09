import tweepy

client = tweepy.Client(
    bearer_token='AAAAAAAAAAAAAAAAAAAAAIYbxwEAAAAA1aRrxoQM4J1zNUdKNOMZ4Jyh9KI%3DO3A019DkeueB8ihb69kcH2dmdcm2p6rdm4X7yP83haioIoEzqW')

query = '#btc -is:retweet lang:en'
tweets = client.search_recent_tweets(query=query, max_results=10)

for tweet in tweets.data:
    print(f"Tweet Text: {tweet.text}")

# Check News API

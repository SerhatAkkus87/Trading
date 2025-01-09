from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='2a7f0b7adbd54c7dbb4bb2437eab53c4')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                          sources='bbc-news,the-verge',
                                          #category='business',
                                          language='en',
                                          #country='us'
                                          )

print(top_headlines)

# /v2/everything
all_articles = newsapi.get_everything(q='bitcoin',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2024-12-19',
                                      to='2025-01-08',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2
                                      )
print(all_articles)

# /v2/top-headlines/sources
sources = newsapi.get_sources()
print(sources)

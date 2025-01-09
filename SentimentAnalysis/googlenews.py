from pygooglenews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import wandb
from tqdm import tqdm

if __name__ == "__main__":
    gn = GoogleNews()

    #top = gn.top_news()
    #print(top)

    CRYPTOs = ['bitcoin -eth']
    headlines = []
    stocks = []
    for crypto in CRYPTOs:
        search = gn.search(crypto)
        for r in search['entries']:
            headlines.append(r['title'])
            stocks.append(crypto)

    print(len(headlines))

    for h in headlines:
        print(h)

    print("Start training...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    print("Training finished...")

    wandb.init(project="CryptoCurrency_Headline_Sentiment_Analysis")
    headlines_table = wandb.Table(columns=["Headline", "Crypto", "Positive", "Negative", "Neutral"])


    def chunkList(list, n):
        for i in range(0, len(list), n):
            yield list[i:i+n]


    chunk_size = 100
    model.eval()
    n = 0

    for lines, cryptos in zip(chunkList(headlines, chunk_size), chunkList(stocks, chunk_size)):
        input = tokenizer(lines, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**input)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        print(f"{n+1}/{int(len(headlines) / chunk_size)}")

        for headline, crypto, pos, neg, neutr in zip(lines, cryptos, prediction[:, 0].tolist(), prediction[:, 1].tolist(), prediction[:, 2].tolist()):
            headlines_table.add_data(headline, crypto, pos, neg, neutr)

        n = n + 1


    wandb.run.log({"Cryptocurrency Sentiment Analysis Table": headlines_table})
    wandb.run.finish()

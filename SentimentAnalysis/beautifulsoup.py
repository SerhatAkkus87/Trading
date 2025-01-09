import beautifulsoup
import requests
from bs4 import BeautifulSoup

url = 'https://www.moneycontrol.com/cryptocurrency/'
response = requests.get(url)
print(response.text)

soup = BeautifulSoup(response.text, 'html.parser')
headlines = soup.find('body').find_all({'h3', 'h2'})

for x in headlines:
    print(x.text.strip())
#86c7d85607e690addff6cea57c058cd0b78bd381
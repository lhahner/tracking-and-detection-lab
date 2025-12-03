import requests
from bs4 import BeautifulSoup
# TODO check scrapy

search_queries = {
        "KITTI Dataset",
        "KITTI Dataset alternatives",
        "Computer Vision dataset",
        "Arial Dataset for computer vision",
        "Urban LIDAR Dataset",
        "Urban RADAR Dataset",
        "Urban Computer Vision Dataset"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:84.0) Gecko/20100101 Firefox/84.0",
}

search_results = []
for query in search_queries:
    print(f"[-] Query \"{query}\" running...")
    page = requests.get('https://html.duckduckgo.com/html/', params = {"q": query}, headers=headers)
    print(f"[--] Visited {page.url}")
    print(f"[--] Got {page}")
    soup = BeautifulSoup(page.text, 'html.parser').find_all("a", class_="result__url", href=True)
    results = []
    for link in soup:
        results.append(link['href'])

    print(f"[-] Found {len(results)} results for query {query}")
    search_results.append(search_results)

print("Finished.")
print(search_results)



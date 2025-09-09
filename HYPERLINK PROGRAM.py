Perform 6 hyperlink-level navigation using browser – use AI on your academic use
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin


# Custom HTML parser to extract <a href="...">
class WikiLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.found_links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for (attr, value) in attrs:
                if attr == "href":
                    self.found_links.append(value)


# Function to grab the first Wikipedia-style link from a page
def fetch_first_wiki_link(page_url):
    try:
        with urllib.request.urlopen(page_url) as resp:
            html_content = resp.read().decode()
            parser = WikiLinkParser()
            parser.feed(html_content)

            for link in parser.found_links:
                abs_url = urljoin(page_url, link)
                if abs_url.startswith("http") and "wiki" in abs_url:
                    return abs_url
    except Exception as e:
        print(f"Error fetching {page_url}: {e}")
        return None
    return None


# Starting Wikipedia page
start_page = "https://en.wikipedia.org/wiki/Machine_learning"
path = [start_page]

# Traverse 6 steps by following first Wiki link
for _ in range(6):
    next_page = fetch_first_wiki_link(path[-1])
    if not next_page:
        break
    path.append(next_page)

# Print visited links
for step, link in enumerate(path, start=1):
    print(f"{step}: {link}")



import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import nltk
from textstat import flesch_reading_ease

# تحميل بيانات NLTK (مرة واحدة تكفي)
nltk.download('punkt_tab')

# -------------------------
# إعدادات الملفات
# -------------------------
CSV_FILE = 'devto_articles_features.csv'
CHECKPOINT_FILE = 'devto_checkpoint.txt'

# -------------------------
# إعدادات أخرى
# -------------------------
SITEMAP_URL = "https://dev.to/sitemap-index.xml"
BATCH_SIZE = 50
REQUEST_DELAY = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
}

# -------------------------
# Helper Functions
# -------------------------

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            idx = f.read().strip()
            return int(idx) if idx.isdigit() else 0
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(index))

def fetch_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url, headers=HEADERS)
    response.raise_for_status()
    sitemap_index = ET.fromstring(response.content)
    return [elem.text for elem in sitemap_index.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]

def fetch_article_urls(sitemap_url):
    response = requests.get(sitemap_url, headers=HEADERS)
    response.raise_for_status()
    sitemap = ET.fromstring(response.content)
    return [elem.text for elem in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]

def compute_text_features(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    unique_words = len(set(words))
    lexical_diversity = unique_words / num_words if num_words > 0 else 0
    readability = flesch_reading_ease(text) if num_words > 0 else 0
    return num_words, num_sentences, avg_sentence_length, lexical_diversity, readability

def scrape_article(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ""

        body_tag = soup.find('div', class_='crayons-article__body')
        body = body_tag.get_text(separator=' ', strip=True) if body_tag else ""

        num_words, num_sentences, avg_sentence_length, lexical_diversity, readability = compute_text_features(body)

        tags = [tag.get_text(strip=True) for tag in soup.find_all('a', class_='crayons-article__tag')]

        author_tag = soup.find('a', class_='crayons-article__subheader__link')
        author = author_tag.get_text(strip=True) if author_tag else ""

        time_tag = soup.find('time')
        published_at = time_tag['datetime'] if time_tag and 'datetime' in time_tag.attrs else ""

        likes_tag = soup.find('button', class_='js-react-button')
        likes_count = int(likes_tag.get_text(strip=True)) if likes_tag and likes_tag.get_text(strip=True).isdigit() else 0

        comments_tag = soup.find('a', class_='crayons-article__comment-count')
        comments_count = int(comments_tag.get_text(strip=True)) if comments_tag and comments_tag.get_text(strip=True).isdigit() else 0

        code_count = len(soup.find_all('pre'))
        images_count = len(soup.find_all('img'))
        num_h2 = len(soup.find_all('h2'))
        num_h3 = len(soup.find_all('h3'))
        num_lists = len(soup.find_all(['ul','ol']))
        num_links = len(soup.find_all('a'))

        return {
            "url": url,
            "title": title,
            "body": body,
            "tags": ", ".join(tags),
            "author": author,
            "published_at": published_at,
            "likes_count": likes_count,
            "comments_count": comments_count,
            "code_snippets_count": code_count,
            "images_count": images_count,
            "num_words": num_words,
            "num_sentences": num_sentences,
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "readability_score": readability,
            "num_h2": num_h2,
            "num_h3": num_h3,
            "num_lists": num_lists,
            "num_links": num_links
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# -------------------------
# MAIN SCRAPING LOGIC
# -------------------------

def main():
    last_index = load_checkpoint()
    print(f"Resuming from sitemap index: {last_index}")

    if os.path.exists(CSV_FILE):
        existing_df = pd.read_csv(CSV_FILE)
        existing_urls = set(existing_df['url'].tolist())
    else:
        existing_urls = set()

    sitemap_urls = fetch_sitemap_urls(SITEMAP_URL)
    print(f"Found {len(sitemap_urls)} sitemaps.")

    for sitemap_idx in range(last_index, len(sitemap_urls)):
        sitemap_url = sitemap_urls[sitemap_idx]
        print(f"Processing sitemap {sitemap_idx + 1}/{len(sitemap_urls)}: {sitemap_url}")

        article_urls = fetch_article_urls(sitemap_url)
        print(f"Found {len(article_urls)} articles in sitemap.")

        articles_data = []

        for i in tqdm(range(0, len(article_urls), BATCH_SIZE), desc="Scraping articles"):
            batch_urls = article_urls[i:i+BATCH_SIZE]
            for url in batch_urls:
                if url in existing_urls:
                    continue
                data = scrape_article(url)
                if data:
                    articles_data.append(data)
                    existing_urls.add(url)
                time.sleep(REQUEST_DELAY)

            if articles_data:
                df = pd.DataFrame(articles_data)
                if not os.path.exists(CSV_FILE):
                    df.to_csv(CSV_FILE, index=False, mode='w', encoding='utf-8-sig')
                else:
                    df.to_csv(CSV_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')
                articles_data = []

        save_checkpoint(sitemap_idx)
        print(f"Checkpoint updated: {sitemap_idx}")

    print("Scraping completed successfully.")

if __name__ == "__main__":
    main()

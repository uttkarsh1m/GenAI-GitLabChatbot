"""
GitLab Handbook & Direction Page Scraper
Retrieves and structures content from GitLab's public pages.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITLAB_SOURCES = {
    "handbook": [
        # Company & Culture
        "https://handbook.gitlab.com/handbook/values/",
        "https://handbook.gitlab.com/handbook/company/mission/",
        "https://handbook.gitlab.com/handbook/company/culture/",
        "https://handbook.gitlab.com/handbook/company/okrs/",
        "https://handbook.gitlab.com/handbook/company/history/",
        # Engineering
        "https://handbook.gitlab.com/handbook/engineering/",
        "https://handbook.gitlab.com/handbook/engineering/development/",
        "https://handbook.gitlab.com/handbook/engineering/infrastructure/",
        "https://handbook.gitlab.com/handbook/engineering/architecture/",
        # Product
        "https://handbook.gitlab.com/handbook/product/",
        "https://handbook.gitlab.com/handbook/product/product-principles/",
        "https://handbook.gitlab.com/handbook/product/ux/",
        # People & HR
        "https://handbook.gitlab.com/handbook/people-group/",
        "https://handbook.gitlab.com/handbook/hiring/",
        "https://handbook.gitlab.com/handbook/total-rewards/",
        "https://handbook.gitlab.com/handbook/leadership/",
        # Operations
        "https://handbook.gitlab.com/handbook/marketing/",
        "https://handbook.gitlab.com/handbook/sales/",
        "https://handbook.gitlab.com/handbook/finance/",
        "https://handbook.gitlab.com/handbook/legal/",
        "https://handbook.gitlab.com/handbook/security/",
        "https://handbook.gitlab.com/handbook/support/",
        # Communication & Process
        "https://handbook.gitlab.com/handbook/communication/",
        "https://handbook.gitlab.com/handbook/it/",
        "https://handbook.gitlab.com/handbook/business-technology/",
        "https://handbook.gitlab.com/handbook/customer-success/",
        "https://handbook.gitlab.com/handbook/alliances/",
    ],
    "direction": [
        "https://about.gitlab.com/direction/",
        "https://about.gitlab.com/direction/dev/",
        "https://about.gitlab.com/direction/ops/",
        "https://about.gitlab.com/direction/sec/",
        "https://about.gitlab.com/direction/data-stores/",
        "https://about.gitlab.com/direction/ai-ml/",
        "https://about.gitlab.com/direction/modelops/",
        "https://about.gitlab.com/direction/analytics/",
        "https://about.gitlab.com/direction/verify/",
        "https://about.gitlab.com/direction/plan/",
    ]
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GitLabHandbookBot/1.0; educational project)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 7]  # seconds between retries


def clean_text(text: str) -> str:
    """Clean and normalize scraped text."""
    if not text:
        return ""
    lines = text.split('\n')
    cleaned = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
    return ' '.join(cleaned)


def scrape_page(url: str, category: str) -> Optional[Dict]:
    """Scrape a single page with retry logic and return structured content."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Scraping: {url} (attempt {attempt + 1})")
            response = requests.get(url, headers=HEADERS, timeout=15,
                                    allow_redirects=True)

            # Skip pages that redirect to login
            if "sign_in" in response.url or "login" in response.url:
                logger.warning(f"Skipped (requires login): {url}")
                return None

            response.raise_for_status()

            # Detect encoding
            if response.encoding and response.encoding.lower() != 'utf-8':
                text_content = response.content.decode(response.encoding, errors='replace')
            else:
                text_content = response.content.decode('utf-8', errors='replace')

            soup = BeautifulSoup(text_content, 'lxml')

            # Remove noise elements
            for tag in soup.find_all(['nav', 'footer', 'script', 'style', 'header',
                                       'aside', 'form', 'button', 'iframe', 'noscript']):
                tag.decompose()

            # Extract title safely
            title = ""
            h1 = soup.find('h1')
            title_tag = soup.find('title')
            if h1:
                title = h1.get_text(strip=True)[:200]
            elif title_tag:
                title = title_tag.get_text(strip=True)[:200]

            if not title:
                title = url.split('/')[-2] or "GitLab Page"

            # Extract main content — try multiple selectors
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('div', class_='main-content') or
                soup.find('div', class_='handbook-content') or
                soup.find('div', id='content') or
                soup.find('body')
            )

            if not main_content:
                logger.warning(f"No main content found for: {url}")
                return None

            # Extract sections with headings
            sections = []
            current_section = {"heading": title, "content": "", "level": 1}

            for element in main_content.find_all(
                ['h1', 'h2', 'h3', 'h4', 'p', 'li', 'blockquote']
            ):
                tag_name = element.name
                if tag_name in ['h1', 'h2', 'h3', 'h4']:
                    if current_section["content"].strip():
                        sections.append(current_section.copy())
                    level = int(tag_name[1])
                    current_section = {
                        "heading": element.get_text(strip=True)[:200],
                        "content": "",
                        "level": level
                    }
                else:
                    text = element.get_text(strip=True)
                    if text and len(text) > 5:
                        current_section["content"] += " " + text

            if current_section["content"].strip():
                sections.append(current_section)

            full_text = clean_text(main_content.get_text())
            # Safe UTF-8 slice — avoid cutting mid-character
            full_text = full_text[:10000]

            return {
                "url": url,
                "title": title,
                "category": category,
                "sections": sections,
                "full_text": full_text,
            }

        except requests.exceptions.Timeout:
            last_error = f"Timeout on attempt {attempt + 1}"
            logger.warning(f"{last_error}: {url}")
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            logger.warning(f"{last_error}: {url}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            logger.warning(f"HTTP {status} for {url}: {e}")
            # Don't retry 4xx errors (except 429)
            if e.response and e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', RETRY_BACKOFF[attempt]))
                logger.info(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
            elif e.response and 400 <= e.response.status_code < 500:
                return None  # No point retrying client errors
            last_error = str(e)
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error scraping {url}: {e}")

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF[attempt]
            logger.info(f"Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"Failed after {MAX_RETRIES} attempts: {url}. Last error: {last_error}")
    return None


def scrape_all_sources(output_path: str = "data/gitlab_content.json") -> List[Dict]:
    """Scrape all GitLab sources and save to JSON."""
    all_documents = []

    for category, urls in GITLAB_SOURCES.items():
        logger.info(f"\n=== Scraping {category} pages ===")
        for url in urls:
            doc = scrape_page(url, category)
            if doc:
                all_documents.append(doc)
            time.sleep(0.5)

    if not all_documents:
        logger.error("No documents scraped! Check network connectivity.")
        return []

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Scraped {len(all_documents)} pages. Saved to {output_path}")
    except OSError as e:
        logger.error(f"Failed to save scraped data: {e}")

    return all_documents


def load_scraped_data(path: str = "data/gitlab_content.json") -> List[Dict]:
    """Load previously scraped data, or scrape fresh if not found."""
    if not os.path.exists(path):
        logger.info("No cached data found. Scraping now...")
        return scrape_all_sources(path)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            logger.warning("Cached data is empty. Re-scraping...")
            return scrape_all_sources(path)
        logger.info(f"Loaded {len(data)} documents from cache.")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load cached data ({e}). Re-scraping...")
        return scrape_all_sources(path)


if __name__ == "__main__":
    docs = scrape_all_sources()
    print(f"Total documents scraped: {len(docs)}")

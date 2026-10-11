"""
Corpus fetching script: downloads the body text of a set of Wikipedia articles and saves
each as data/docs/*.txt, to be used as the RAG knowledge base.

Source: Wikipedia. The content is licensed under CC BY-SA, which allows redistribution
with attribution, so it can be published together with the project. The topics are
general knowledge (geography, history, basic science, nature, inventions, people, ...);
each article covers one clear topic, which makes it easy to build a retrieval evaluation set.

It uses the official Wikipedia API (a plain public web interface, no account or key
needed) through requests and gets plain-text bodies (wiki markup, images and citation
markers are already stripped).

Usage (after `conda activate agent-rag`, from the project root):
    python src/fetch_wikipedia_corpus.py
"""

import re
import time
from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
# The Wikipedia API requires a User-Agent that identifies the client; requests without
# one are treated as unidentified bots and rejected with 403.
HEADERS = {"User-Agent": "agent-rag-assistant-demo/1.0 (student learning project, not for production use)"}

TOPICS = [
    # Geography
    "Mount Everest", "Amazon rainforest", "Sahara", "Nile", "Great Barrier Reef",
    "Grand Canyon", "Antarctica", "Pacific Ocean", "Great Wall of China", "Eiffel Tower",
    "Statue of Liberty", "Colosseum", "Taj Mahal", "Machu Picchu", "Mount Fuji",
    # History
    "Ancient Egypt", "Roman Empire", "World War II", "World War I", "French Revolution",
    "Industrial Revolution", "Renaissance", "Cold War", "Great Depression", "Silk Road",
    # Basic science
    "Solar System", "Black hole", "Gravity", "Photosynthesis", "DNA",
    "Evolution", "Periodic table", "Big Bang", "Speed of light", "Electricity",
    "Nuclear fusion", "Plate tectonics", "Volcano", "Earthquake", "Climate change",
    # Nature / biology
    "Human brain", "Immune system", "Blue whale", "Giant panda", "Coral reef",
    "Bee", "Shark", "Dinosaur", "Amazon rainforest",
    # Technology and inventions
    "Internet", "Printing press", "Telephone", "Airplane", "Automobile",
    "Steam engine", "Vaccine", "Antibiotic", "Nuclear power", "Renewable energy",
    # Culture / people
    "Olympic Games", "World Health Organization", "United Nations", "Nobel Prize",
    "Leonardo da Vinci", "Albert Einstein", "Isaac Newton", "Charles Darwin",
    "William Shakespeare", "Mona Lisa",
]
TOPICS = sorted(set(TOPICS))  # de-duplicate (e.g. "Amazon rainforest" is listed twice)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "docs"


def safe_filename(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", title)


def fetch_article(title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        return page.get("extract", "").strip() or None
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []

    for title in TOPICS:
        out_path = OUTPUT_DIR / f"{safe_filename(title)}.txt"
        if out_path.exists():
            ok += 1
            continue  # already fetched in a previous run, skip

        print(f"Fetching: {title} ...", end=" ", flush=True)

        text = None
        for attempt in range(4):
            try:
                text = fetch_article(title)
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"Rate limited, retrying in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                    continue
                print(f"Network error, skipping ({e})")
                break
            except requests.RequestException as e:
                print(f"Network error, skipping ({e})")
                break

        if not text:
            print("Not found or could not be fetched, skipping")
            failed.append(title)
            continue

        out_path.write_text(text, encoding="utf-8")
        print(f"done ({len(text)} chars)")
        ok += 1
        time.sleep(1)  # be gentle with the API; 0.2 s between requests got rate limited

    print(f"\nSucceeded: {ok}, failed/skipped: {len(failed)}")
    if failed:
        print("Failed list:", failed)


if __name__ == "__main__":
    main()

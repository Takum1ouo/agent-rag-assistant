# Corpus

This directory holds the knowledge-base corpus. Both `.pdf` and `.txt` files are supported; `python src/ingest.py` reads every PDF and txt file here and builds the index.

## Current corpus

The body text of 68 English Wikipedia articles (txt) on general-knowledge topics, fetched through the official Wikipedia API with `python src/fetch_wikipedia_corpus.py`.

The text comes from [Wikipedia](https://www.wikipedia.org/) and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each file name is the title of the corresponding Wikipedia article; the original text and its authorship history are available on the article's page at `https://en.wikipedia.org/wiki/<article title>`.

## Using your own corpus

Put PDF or txt files in this directory and re-run `python src/ingest.py` to rebuild the index.

Note: `.gitignore` ignores `*.pdf` files in this directory by default (to avoid accidentally committing copyrighted material); txt files are tracked by git.

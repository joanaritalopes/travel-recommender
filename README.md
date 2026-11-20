# Travel Recommender

A city recommender that suggests travel destinations based on what a person enjoys, using NLP on travel-guide text. Each city gets a profile built from its Wikivoyage article: a sentence embedding that captures what the place is like, and activity tags (beaches, nightlife, museums, hiking) that explain why it was recommended.

## How it works

1. **Candidate cities** are selected from GeoNames by population, with a per-country cap to keep the list geographically diverse.
2. **Wikivoyage articles** are collected for each candidate through the MediaWiki API, resolving redirects and skipping disambiguation pages.
3. **Text cleaning** keeps only the sections that describe a city's character (see design decisions below).
4. **Destination embeddings** turn each city's text into a vector with a sentence-transformer model.
5. **Activity tags** are assigned with zero-shot classification, so every recommendation comes with a reason.
6. **Recommendations** rank cities by similarity between a user's preferences and each city profile.

A second phase will add collaborative filtering on Booking.com's multi-destination trip dataset, learning from real trip sequences.

## Design decisions

### Candidate selection: population is a starting point, not a filter
The largest cities by population are not necessarily the most visited destinations. GeoNames provides a broad candidate list, and the quality of each city's Wikivoyage article then decides whether it has enough content to describe. Districts of larger cities (GeoNames feature code `PPLX`) are excluded so they don't compete with the city they belong to.

### Collecting data through an API, not scraping
Article text comes from the MediaWiki API as structured JSON. Page metadata is requested in batches of 50 titles, while full text is fetched per page. Requests identify the project with a `User-Agent` header and pause between calls, following Wikimedia's API guidelines.

### Section selection as feature selection on text
Wikivoyage city articles mix two kinds of content: sections describing a city's character (Understand, See, Do, Buy, Eat, Drink) and sections about logistics (Get in, Get around, Sleep, Stay safe, Connect). Every city has an airport, hotels and safety advice, so logistics text makes cities look more similar to each other in embedding space, which is the opposite of what a recommender needs. Only the character sections and the introduction are kept. Sections are stored separately, so later steps can use them individually, for example tagging activities from See and Do. Articles with fewer than 300 words after cleaning are dropped, since that is too little text to describe a city.

### Section-balanced embeddings for long texts
City profiles are embedded with `all-MiniLM-L6-v2`, a compact sentence-transformer that runs on a laptop. The model reads at most 256 tokens, while the median city text is about 1,500 words, so embedding the full text directly would only capture its opening. Each section is split into 150-word chunks, every chunk is embedded, chunks are averaged into section vectors, and section vectors are averaged into one city vector. Averaging per section first gives every section an equal say: otherwise long sections, such as Eat pages listing dozens of restaurants, would dominate a city's profile. City vectors are normalized to unit length, so cosine similarity reduces to a dot product.

## Project structure

```
src/travel_recsys/
├── data/         collecting and cleaning data
├── features/     embeddings and activity tags
├── models/       content-based and collaborative recommenders
└── evaluation/   metrics and test personas
scripts/          pipeline entry points
notebooks/        exploration only
```

## Setup

```bash
python3 -m venv travenv
source travenv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running the pipeline

```bash
python -m travel_recsys.data.01_cities
python -m travel_recsys.data.01_wikivoyage
python -m travel_recsys.data.03_clean_text
python -m travel_recsys.features.04_embeddings
```

## Results

- Wikivoyage pages matched: 2,020
- Cities with usable text after cleaning: 1,656 (364 dropped for having under 300 words)
- Median article length after cleaning: 1,525 words
- City embeddings: 1,656 vectors of 384 dimensions

## Data sources and licenses

- City data from [GeoNames](https://www.geonames.org/), licensed under CC BY 4.0.
- Destination text from [Wikivoyage](https://en.wikivoyage.org/), licensed under CC BY-SA.
- Trip sequences (phase 2) from the [Booking.com Multi-Destination Trips dataset](https://huggingface.co/datasets/Booking-com/multi-destination-trip-dataset).

## Status

- [x] Candidate city selection
- [x] Wikivoyage data collection
- [x] Text cleaning
- [x] Destination embeddings
- [ ] Activity tags  (including budget labels)
- [ ] Content-based recommender
- [ ] Evaluation
- [ ] Collaborative filtering on Booking.com trips
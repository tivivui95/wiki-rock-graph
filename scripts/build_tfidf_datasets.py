#!/usr/bin/env python3
"""Build cached Wikipedia text and TF-IDF datasets for the rock artist graph."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import wikipediaapi

STOPWORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as fh:
        graph = pickle.load(fh)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected NetworkX graph, found {type(graph)!r}")
    return graph


def load_genre_mapping(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    return {k: v for k, v in mapping.items() if v}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
    return [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS]


def build_genre_documents(
    texts: Dict[str, str], genre_mapping: Dict[str, List[str]]
) -> defaultdict[str, List[str]]:
    documents: defaultdict[str, List[str]] = defaultdict(list)
    for artist, genres in genre_mapping.items():
        text = texts.get(artist)
        if not text:
            continue
        tokens = tokenize(text)
        documents[genres[0]].extend(tokens)
    return documents


def build_community_documents(
    texts: Dict[str, str], community_map: Dict[str, int]
) -> defaultdict[int, List[str]]:
    documents: defaultdict[int, List[str]] = defaultdict(list)
    for artist, comm_id in community_map.items():
        text = texts.get(artist)
        if not text:
            continue
        tokens = tokenize(text)
        documents[comm_id].extend(tokens)
    return documents


def compute_tf_idf(documents: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    term_counts = {name: Counter(tokens) for name, tokens in documents.items()}
    vocabulary = {word for tokens in documents.values() for word in tokens}
    doc_count = len(documents)

    idf: Dict[str, float] = {}
    for word in vocabulary:
        containing = sum(1 for tokens in documents.values() if word in tokens)
        idf[word] = math.log((doc_count / containing), 10) if containing else 0.0

    tf_idf: Dict[str, Dict[str, float]] = {}
    for name, counts in term_counts.items():
        total = sum(counts.values()) or 1
        tf_idf[name] = {word: (count / total) * idf[word] for word, count in counts.items()}
    return tf_idf


def fetch_wikipedia_texts(
    artists: Iterable[str],
    sleep: float,
    session: wikipediaapi.Wikipedia,
) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for artist in artists:
        page = session.page(artist.replace(" ", "_"))
        if page.exists():
            texts[artist] = page.text
        time.sleep(sleep)
    return texts


def compute_structural_communities(graph: nx.Graph) -> Tuple[List[List[str]], Dict[str, int]]:
    communities = nx.algorithms.community.louvain_communities(graph, seed=42)
    mapping: Dict[str, int] = {}
    for idx, members in enumerate(communities):
        for node in members:
            mapping[node] = idx
    return [list(comm) for comm in communities], mapping


def save_pickle(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(data, fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("artist_graph.pkl"),
        help="Path to the pickled NetworkX graph.",
    )
    parser.add_argument(
        "--genres-path",
        type=Path,
        default=Path("data/artist_genres.json"),
        help="Path to the cached artist->genres JSON.",
    )
    parser.add_argument(
        "--artist-texts-path",
        type=Path,
        default=Path("data/artist_texts.pkl"),
        help="Where to store cached Wikipedia article texts.",
    )
    parser.add_argument(
        "--tfidf-data-path",
        type=Path,
        default=Path("data/tfidf_data.pkl"),
        help="Where to store the complete TF-IDF dataset.",
    )
    parser.add_argument(
        "--genre-tfidf-path",
        type=Path,
        default=Path("data/genre_tfidf.pkl"),
        help="Where to store genre-only TF-IDF data.",
    )
    parser.add_argument(
        "--community-tfidf-path",
        type=Path,
        default=Path("data/community_tfidf.pkl"),
        help="Where to store community-only TF-IDF data.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Delay between Wikipedia requests (seconds).",
    )
    args = parser.parse_args()

    graph = load_graph(args.graph_path)
    genres = load_genre_mapping(args.genres_path)

    undirected = graph.to_undirected()
    nodes_with_genres = [node for node in undirected.nodes if node in genres]
    subgraph = undirected.subgraph(nodes_with_genres).copy()

    communities, community_map = compute_structural_communities(subgraph)

    wiki = wikipediaapi.Wikipedia("wiki-rock-graph-data (https://github.com/tivivui95/wiki-rock-graph)", "en")
    print(f"Fetching Wikipedia text for {len(nodes_with_genres)} artists...")
    artist_texts = fetch_wikipedia_texts(nodes_with_genres, args.sleep, wiki)
    save_pickle(artist_texts, args.artist_texts_path)
    print(f"Saved artist texts to {args.artist_texts_path}")

    genre_documents = build_genre_documents(artist_texts, genres)
    community_documents = build_community_documents(artist_texts, community_map)

    genre_tfidf = compute_tf_idf(genre_documents)
    community_tfidf = compute_tf_idf(community_documents)

    tfidf_bundle = {
        "genre_documents": genre_documents,
        "genre_tfidf": genre_tfidf,
        "community_documents": community_documents,
        "community_tfidf": community_tfidf,
        "community_map": community_map,
        "structural_communities": communities,
    }

    save_pickle(tfidf_bundle, args.tfidf_data_path)
    save_pickle({"genre_documents": genre_documents, "genre_tfidf": genre_tfidf}, args.genre_tfidf_path)
    save_pickle(
        {
            "community_documents": community_documents,
            "community_tfidf": community_tfidf,
            "community_map": community_map,
            "structural_communities": communities,
        },
        args.community_tfidf_path,
    )

    print("TF-IDF datasets saved successfully.")


if __name__ == "__main__":
    main()

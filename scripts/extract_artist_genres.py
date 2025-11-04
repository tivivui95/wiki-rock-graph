#!/usr/bin/env python3
"""Fetch genre annotations for rock artists from Wikipedia infoboxes."""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda it, **kwargs: it  # type: ignore


def extract_genres_from_infobox(artist_name: str, session: requests.Session) -> List[str]:
    """Extract a list of genres from a Wikipedia infobox."""
    url = f"https://en.wikipedia.org/wiki/{artist_name.replace(' ', '_')}"
    response = session.get(url, headers={"User-Agent": "AssignmentProject/1.0"}, timeout=30)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    infobox = soup.find("table", class_="infobox")
    if not infobox:
        return []

    genres: List[str] = []
    for row in infobox.find_all("tr"):
        header = row.find("th")
        if not header:
            continue

        header_text = header.get_text(strip=True).lower()
        if "genre" not in header_text and "musical style" not in header_text:
            continue

        data_cell = row.find("td")
        if not data_cell:
            continue

        for link in data_cell.find_all("a"):
            genre_text = html.unescape(link.get_text(strip=True)).lower().strip()
            if not _is_valid_genre(genre_text):
                continue
            genres.append(_normalize_genre(genre_text))

        if not genres:
            all_text = html.unescape(data_cell.get_text(separator=" "))
            for candidate in re.split(r"[,;]", all_text):
                genre_text = re.sub(r"\[.*?\]|\(.*?\)", "", candidate).strip().lower()
                if _is_valid_genre(genre_text):
                    genres.append(_normalize_genre(genre_text))
        break

    # Deduplicate while preserving order
    seen = set()
    unique_genres = []
    for genre in genres:
        if genre not in seen:
            seen.add(genre)
            unique_genres.append(genre)
    return unique_genres


def _is_valid_genre(value: str) -> bool:
    if not value:
        return False
    if len(value) < 3 or len(value) > 50:
        return False
    if any(fragment in value for fragment in [".mw-", "hlist", "class=", "<", ">"]):
        return False
    if value.startswith("[") and value.endswith("]"):
        return False
    if value.isdigit():
        return False
    skip_words = {"edit", "citation needed", "note", "see also"}
    return not any(skip in value for skip in skip_words)


def _normalize_genre(value: str) -> str:
    value = value.replace("\u00a0", " ")
    replacements = {
        "rock'n'roll": "rock and roll",
        "rock 'n' roll": "rock and roll",
        "rock & roll": "rock and roll",
        "rhythm and blues": "r&b",
        "rhythm & blues": "r&b",
    }
    for original, replacement in replacements.items():
        value = value.replace(original, replacement)
    return value


def load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def iterate_artists(graph) -> Iterable[str]:
    nodes = graph.nodes() if hasattr(graph, "nodes") else graph
    return list(nodes)


def save_json(data: Dict[str, List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def save_stats(mapping: Dict[str, List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_genres = [genre for genres in mapping.values() for genre in genres]
    stats = {
        "artists_with_genres": len(mapping),
        "total_genre_mentions": len(all_genres),
        "distinct_genres": len(set(all_genres)),
        "average_genres_per_artist": (
            sum(len(genres) for genres in mapping.values()) / len(mapping)
            if mapping
            else 0.0
        ),
        "top_genres": Counter(
            genres[0] for genres in mapping.values() if genres
        ).most_common(20),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=Path("artist_graph.pkl"),
        help="Path to the pickled NetworkX graph.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/artist_genres.json"),
        help="Where to store the JSON mapping of artist -> genres.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/artist_genres_stats.json"),
        help="Where to store summary statistics (JSON).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay (seconds) between requests to respect Wikipedia rate limits.",
    )
    args = parser.parse_args()

    graph = load_graph(args.graph_path)
    artists = iterate_artists(graph)
    mapping: Dict[str, List[str]] = {}

    print(f"Loaded graph with {len(artists)} artists.")
    print("Fetching genres from Wikipedia infoboxes...")

    session = requests.Session()
    for artist in tqdm(artists, desc="Fetching", unit="artist"):
        genres = extract_genres_from_infobox(artist, session)
        if genres:
            mapping[artist] = genres
        time.sleep(args.sleep)

    # Sort for determinism
    mapping = dict(sorted(mapping.items(), key=lambda item: item[0].lower()))
    save_json(mapping, args.output_path)
    save_stats(mapping, args.stats_path)

    print(f"\nStored {len(mapping)} artist entries at {args.output_path}")
    print(f"Summary statistics saved to {args.stats_path}")


if __name__ == "__main__":
    main()

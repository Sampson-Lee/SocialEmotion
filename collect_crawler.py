#!/usr/bin/env python3
"""Scrape YouTube search results without platform APIs.

The crawler builds query strings from social emotion synonyms, optionally
combined with "social anchors" (situations/interactions/media contexts) to
reduce manual filtering. Each retrieved video stores retrieval provenance so
duplicates can be merged without losing which query retrieved them.

Search is performed via Selenium (headless Chrome). Results are saved to JSONL
metadata files with watch URLs for manual review.
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import time
import sys
import urllib.parse
from typing import Collection, Dict, Iterable, List, Optional

EMOTION_SYNONYMS = {
    "embarrassment": [
        "embarrassing",
        "awkward",
        "awkward moment",
        "cringe",
        "cringey",
        "mortifying",
        "secondhand embarrassment",
    ],
    "jealousy": [
        "jealous",
        "jealousy",
        "jealous moment",
        "jealous reaction",
        "she got jealous",
        "he got jealous",
    ],
    "pride": [
        "proud moment",
        "so proud",
        "proud of",
        "public recognition",
        "achievement celebrated",
    ],
    "guilt": [
        "apology",
        "apologize",
        "apologizing",
        "said sorry",
        "public apology",
        "my fault",
    ],
}

SUPPORTED_EMOTIONS = [
    "embarrassment",
    "guilt",
    "jealousy",
    "pride",
]
EMOTION_HELP_TEXT = ", ".join(SUPPORTED_EMOTIONS)

SUPPORTED_ANCHOR_TYPES = [
    "media",
    "situation",
    "interaction",
    "game",
]

INTERACTION_ANCHORS_BY_EMOTION: Dict[str, List[str]] = {
    "embarrassment": [
        "awkward silence",
        "called out",
        "public mistake",
        "put on the spot",
    ],
    "guilt": [
        "apology",
        "apologize",
        "said sorry",
        "my fault",
        "make it up",
        "confession",
    ],
    "jealousy": [
        "flirting",
        "third wheel",
        "ex shows up",
        "relationship drama",
        "jealous reaction",
    ],
    "pride": [
        "congratulations",
        "compliment",
        "praise",
        "proud of you",
        "celebration",
        "recognition",
    ],
}

DEFAULT_ANCHORS_CFG: Dict[str, List[str]] = {
    "media_anchors": [
        "tv show",
        "reality show",
        "interview",
        "podcast",
        "livestream",
    ],
    "situation_anchors": [
        "party",
        "meeting",
        "classroom",
        "workplace",
        "date",
        "family dinner",
    ],
    "game_anchors": [
        "werewolf",
        "mafia game",
        "board game",
        "truth or dare",
        "prank",
        "social experiment",
    ],
}

def sanitize_for_path(value: str, default: str = "emotion") -> str:
    """Return a lowercase filesystem-safe token."""
    base = (value or "").strip().lower()
    if not base:
        return default
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in base)
    sanitized = sanitized.strip("_")
    return sanitized or default


def dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        normalized = (value or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def resolve_emotions(
    emotions: Optional[List[str]],
    all_emotions: bool,
) -> List[str]:
    candidates: List[str] = []
    if all_emotions:
        candidates.extend(SUPPORTED_EMOTIONS)
    if emotions:
        candidates.extend(emotions)
    resolved = dedupe_preserve(candidates)
    if not resolved:
        raise SystemExit(
            "Provide --emotions or --all-emotions so the crawler knows which feelings to target."
        )
    canonical: List[str] = []
    seen = set()
    unsupported: List[str] = []
    for item in resolved:
        key = (item or "").strip().lower()
        if not key:
            continue
        if key not in SUPPORTED_EMOTIONS:
            unsupported.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        canonical.append(key)
    if unsupported:
        raise SystemExit(
            f"Unsupported emotion(s): {', '.join(unsupported)}. Supported: {EMOTION_HELP_TEXT}."
        )
    if not canonical:
        raise SystemExit(f"No supported emotions selected. Supported: {EMOTION_HELP_TEXT}.")
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect candidate social-emotion clips via Selenium (no API key needed)."
    )
    emotion_group = parser.add_mutually_exclusive_group(required=True)
    emotion_group.add_argument(
        "--emotions",
        nargs="+",
        help=f"One or more emotions to collect (choices: {EMOTION_HELP_TEXT}).",
    )
    emotion_group.add_argument(
        "--all-emotions",
        action="store_true",
        help="Collect metadata for every supported emotion.",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=500,
        help="Maximum results to fetch per query.",
    )
    parser.add_argument(
        "--max-queries-per-emotion",
        type=int,
        default=200,
        help="Safety cap on queries processed per emotion (adaptive stop may finish earlier).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Deterministically shuffle query order (and sampling when capped).",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=int,
        help="Skip videos shorter than this (useful to drop shorts).",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        default=180,
        help="Skip videos longer than this (default 180 seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data") / "youtube_crawler_candidates",
        help=(
            "Base directory for metadata; by default a per-run subfolder is created inside it."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional per-run subdirectory name under --output-dir (default: UTC timestamp).",
    )
    parser.add_argument(
        "--run-subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write outputs into a per-run subdirectory under --output-dir (default true).",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run chrome in headless mode (default true).",
    )
    parser.add_argument(
        "--scroll-pause-seconds",
        type=float,
        default=1.0,
        help="Pause between scrolls when using selenium (default 1.0).",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=15,
        help="Safety cap on scrolls per query when using selenium (default 15).",
    )
    parser.add_argument(
        "--min-scroll-novelty",
        type=float,
        default=0.10,
        help="Stop scrolling when novelty rate falls below this (default 0.10). Set 0 to disable.",
    )
    parser.add_argument(
        "--scroll-novelty-patience",
        type=int,
        default=2,
        help="Consecutive low-novelty scrolls before stopping (default 2).",
    )
    parser.add_argument(
        "--min-new-videos-per-query",
        type=int,
        default=10,
        help="Stop exploring new queries when each adds fewer than this many new videos (default 10). Set 0 to disable.",
    )
    parser.add_argument(
        "--query-patience",
        type=int,
        default=3,
        help="Consecutive low-gain queries before stopping an emotion (default 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without hitting the network.",
    )
    return parser.parse_args()


ANCHOR_TYPE_TO_CFG_KEY = {
    "media": "media_anchors",
    "situation": "situation_anchors",
    "interaction": "interaction_anchors_by_emotion",
    "game": "game_anchors",
}


def default_anchors_cfg() -> Dict[str, object]:
    cfg: Dict[str, object] = {key: list(values) for key, values in DEFAULT_ANCHORS_CFG.items()}
    cfg["interaction_anchors_by_emotion"] = {
        emotion: list(anchors) for emotion, anchors in INTERACTION_ANCHORS_BY_EMOTION.items()
    }
    return cfg


def stable_int_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_queries(
    emotions: List[str],
    anchors_cfg: Dict,
    *,
    max_queries_per_emotion: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Build retrieval queries for each emotion.

    Output is a list of dictionaries with fields:
      - query: final query string
      - emotion: canonical emotion label (key in EMOTION_SYNONYMS)
      - emotion_synonym: synonym used
      - retrieval_mode: emotion_plus_anchor
      - anchor_type: media | situation | interaction | game
      - anchor: anchor string (or None)
    """
    selected_anchor_types = list(SUPPORTED_ANCHOR_TYPES)

    if max_queries_per_emotion is not None and max_queries_per_emotion < 1:
        raise SystemExit("--max-queries-per-emotion must be >= 1.")

    all_specs: List[Dict] = []
    for emotion in emotions:
        canonical = (emotion or "").strip().lower()
        if not canonical:
            continue
        if canonical not in EMOTION_SYNONYMS:
            raise SystemExit(
                f"Unsupported emotion '{emotion}'. Supported: {', '.join(SUPPORTED_EMOTIONS)}."
            )

        synonyms = EMOTION_SYNONYMS.get(canonical) or []
        per_emotion: List[Dict] = []
        seen: set = set()

        def add(spec: Dict) -> None:
            key = json.dumps(spec, sort_keys=True, ensure_ascii=False)
            if key in seen:
                return
            seen.add(key)
            per_emotion.append(spec)

        for synonym in synonyms:
            synonym = (synonym or "").strip()
            if not synonym:
                continue
            synonym_norm = " ".join(synonym.lower().split())
            for anchor_type in selected_anchor_types:
                cfg_key = ANCHOR_TYPE_TO_CFG_KEY[anchor_type]
                if anchor_type == "interaction":
                    anchors_by_emotion = anchors_cfg.get(cfg_key)
                    if isinstance(anchors_by_emotion, dict):
                        emotion_anchors = anchors_by_emotion.get(canonical)
                        anchors = emotion_anchors if isinstance(emotion_anchors, list) else []
                    else:
                        anchors = []
                else:
                    anchors = anchors_cfg.get(cfg_key) or []
                if not isinstance(anchors, list):
                    anchors = []
                for anchor in anchors:
                    anchor = (anchor or "").strip()
                    if not anchor:
                        continue
                    anchor_norm = " ".join(anchor.lower().split())
                    if anchor_norm and anchor_norm in synonym_norm:
                        continue
                    query = f"{synonym} {anchor}".strip()
                    add(
                        {
                            "query": query,
                            "emotion": canonical,
                            "emotion_synonym": synonym,
                            "retrieval_mode": "emotion_plus_anchor",
                            "anchor_type": anchor_type,
                            "anchor": anchor,
                        }
                    )

        if seed is not None:
            rng = random.Random(stable_int_seed(seed, canonical))
            rng.shuffle(per_emotion)
        else:
            random.shuffle(per_emotion)

        if max_queries_per_emotion is not None:
            per_emotion = per_emotion[:max_queries_per_emotion]

        all_specs.extend(per_emotion)

    return all_specs


def make_provenance_entry(spec: Dict) -> Dict:
    return {
        "query": spec.get("query"),
        "emotion": spec.get("emotion"),
        "emotion_synonym": spec.get("emotion_synonym"),
        "retrieval_mode": spec.get("retrieval_mode"),
        "anchor_type": spec.get("anchor_type"),
        "anchor": spec.get("anchor"),
    }


def seconds_to_iso8601(seconds: Optional[float]) -> Optional[str]:
    """Convert seconds to an ISO 8601 duration string like PT3M20S."""
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    parts: List[str] = ["P"]
    if days:
        parts.append(f"{days}D")
    parts.append("T")
    if hrs:
        parts.append(f"{hrs}H")
    if mins:
        parts.append(f"{mins}M")
    parts.append(f"{sec}S")
    return "".join(parts)


def default_run_id(now: Optional[dt.datetime] = None) -> str:
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    else:
        now = now.astimezone(dt.timezone.utc)
    return now.strftime("%Y%m%d_%H%M%SZ")


def create_unique_run_dir(output_root: pathlib.Path, run_id: str) -> pathlib.Path:
    token = sanitize_for_path(run_id, default="run")
    suffix = 0
    while True:
        candidate = output_root / (token if suffix == 0 else f"{token}_{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1




def selenium_available() -> bool:
    try:
        import selenium  # noqa: F401
    except ImportError:
        return False
    return True


def parse_duration_to_seconds(value: str) -> Optional[int]:
    text = (value or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in {"LIVE", "PREMIERE"}:
        return None
    parts = [p for p in text.split(":") if p.strip()]
    if not parts or any(not p.isdigit() for p in parts):
        return None
    total = 0
    for part in parts:
        total = total * 60 + int(part)
    return total


def extract_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.path == "/watch":
        qs = urllib.parse.parse_qs(parsed.query)
        return (qs.get("v") or [None])[0]
    if parsed.path.startswith("/shorts/"):
        candidate = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        return candidate or None
    return None


def build_youtube_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)


def create_selenium_driver(headless: bool):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
    except ImportError as exc:
        raise SystemExit(
            "selenium is required for the selenium search backend. Install with: pip install selenium"
        ) from exc

    options = ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)


def search_youtube_selenium(
    driver,
    query: str,
    max_results: int,
    scroll_pause_seconds: float,
    max_scrolls: int,
    *,
    known_video_ids: Optional[Collection[str]] = None,
    min_scroll_novelty: float = 0.10,
    scroll_novelty_patience: int = 2,
    debug: Optional[Dict] = None,
) -> List[Dict]:
    """Return video metadata dictionaries for a single query via Selenium-rendered YouTube search."""
    from selenium.common.exceptions import NoSuchElementException, TimeoutException  # type: ignore
    from selenium.webdriver.common.by import By  # type: ignore
    from selenium.webdriver.support import expected_conditions as EC  # type: ignore
    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore

    url = build_youtube_search_url(query)
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer"))
        )
    except TimeoutException:
        return []

    results: List[Dict] = []
    seen_ids: set = set()

    scrolls_done = 0
    low_novelty_streak = 0
    novelty_stats: List[Dict] = []
    stop_reason = "unknown"
    known_ids = known_video_ids or ()

    while len(seen_ids) < max_results:
        start_count = len(seen_ids)
        renderers = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        for renderer in renderers:
            try:
                link = renderer.find_element(By.CSS_SELECTOR, "a#video-title")
            except NoSuchElementException:
                continue
            href = (link.get_attribute("href") or "").strip()
            video_id = extract_video_id(href)
            if not video_id or video_id in seen_ids:
                continue

            title = (link.get_attribute("title") or link.text or "").strip()

            channel_title = None
            channel_id = None
            try:
                channel_link = renderer.find_element(By.CSS_SELECTOR, "ytd-channel-name a")
                channel_title = (channel_link.text or "").strip() or None
                channel_href = (channel_link.get_attribute("href") or "").strip()
                parsed = urllib.parse.urlparse(channel_href)
                if "/channel/" in parsed.path:
                    channel_id = parsed.path.split("/channel/", 1)[1].split("/", 1)[0] or None
            except NoSuchElementException:
                pass

            duration_text = ""
            try:
                duration_el = renderer.find_element(
                    By.CSS_SELECTOR, "ytd-thumbnail-overlay-time-status-renderer span#text"
                )
                duration_text = (duration_el.text or "").strip()
            except NoSuchElementException:
                duration_text = ""
            duration_seconds = parse_duration_to_seconds(duration_text)

            description = None
            try:
                desc_el = renderer.find_element(By.CSS_SELECTOR, "yt-formatted-string#description-text")
                description = (desc_el.text or "").strip() or None
            except NoSuchElementException:
                pass

            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            results.append(
                {
                    "video_id": video_id,
                    "title": title or None,
                    "channel_id": channel_id,
                    "description": description,
                    "channel_title": channel_title,
                    "duration_iso": seconds_to_iso8601(duration_seconds),
                    "duration_seconds": duration_seconds,
                    "like_count": None,
                    "query": query,
                    "watch_url": watch_url,
                }
            )
            seen_ids.add(video_id)
            if len(seen_ids) >= max_results:
                break

        new_ids_this_batch = len(seen_ids) - start_count
        new_overall_this_batch = 0
        if new_ids_this_batch:
            for item in results[-new_ids_this_batch:]:
                video_id = item.get("video_id")
                if video_id and video_id not in known_ids:
                    new_overall_this_batch += 1
        novelty = new_overall_this_batch / max(new_ids_this_batch, 1)
        novelty_stats.append(
            {
                "scroll_index": scrolls_done,
                "fetched": new_ids_this_batch,
                "new_overall": new_overall_this_batch,
                "novelty": novelty,
                "unique_so_far": len(seen_ids),
            }
        )

        if novelty < min_scroll_novelty:
            low_novelty_streak += 1
        else:
            low_novelty_streak = 0

        if len(seen_ids) >= max_results:
            stop_reason = "max_results"
            break
        if scrolls_done >= max_scrolls:
            stop_reason = "max_scrolls_cap"
            break
        if low_novelty_streak >= scroll_novelty_patience:
            stop_reason = "low_novelty"
            break

        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(max(0.1, scroll_pause_seconds))
        scrolls_done += 1

    if stop_reason == "unknown":
        stop_reason = "exhausted"
    if debug is not None:
        debug.update(
            {
                "stop_reason": stop_reason,
                "scrolls_executed": scrolls_done,
                "max_scrolls_cap": max_scrolls,
                "min_scroll_novelty": min_scroll_novelty,
                "scroll_novelty_patience": scroll_novelty_patience,
                "novelty_stats": novelty_stats,
                "result_count": len(results),
            }
        )
    return results


def filter_results(
    results: Iterable[Dict],
    min_duration_seconds: Optional[int],
    max_duration_seconds: Optional[int],
) -> List[Dict]:
    """Filter raw search results for duration constraints before writing metadata."""
    filtered: List[Dict] = []
    for item in results:
        duration_val = item.get("duration_seconds")
        if duration_val is not None:
            if min_duration_seconds is not None and duration_val < min_duration_seconds:
                continue
            if max_duration_seconds is not None and duration_val > max_duration_seconds:
                continue

        filtered.append(item)
    return filtered


def provenance_key(entry: Dict) -> str:
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def merge_provenance(existing: List[Dict], new_entry: Dict) -> bool:
    """Append new provenance entry if not already present."""
    if not existing:
        existing.append(new_entry)
        return True
    new_key = provenance_key(new_entry)
    existing_keys = {provenance_key(entry) for entry in existing}
    if new_key in existing_keys:
        return False
    existing.append(new_entry)
    return True


def upsert_video_record(
    records_by_id: Dict[str, Dict],
    item: Dict,
    provenance_entry: Dict,
) -> bool:
    """Insert or merge a video record keyed by video_id.

    Returns True when inserted, False when merged into an existing record.
    """
    video_id = (item.get("video_id") or "").strip()
    if not video_id:
        return False

    existing = records_by_id.get(video_id)
    if existing is None:
        record = dict(item)
        record["retrieval_provenance"] = [provenance_entry]
        records_by_id[video_id] = record
        return True

    for key, value in item.items():
        if key == "retrieval_provenance":
            continue
        if existing.get(key) is None and value is not None:
            existing[key] = value

    provenance_list = existing.get("retrieval_provenance")
    if not isinstance(provenance_list, list):
        provenance_list = []
        existing["retrieval_provenance"] = provenance_list
    merge_provenance(provenance_list, provenance_entry)
    return False


def summarize_retrieval(records: List[Dict], query_specs: List[Dict]) -> Dict:
    """Summarize unique videos by retrieval_mode and anchor_type."""
    query_counts_by_mode: Dict[str, int] = {}
    query_counts_by_anchor_type: Dict[str, int] = {}
    for spec in query_specs:
        mode = spec.get("retrieval_mode") or "unknown"
        anchor_type = spec.get("anchor_type") or "unknown"
        query_counts_by_mode[mode] = query_counts_by_mode.get(mode, 0) + 1
        query_counts_by_anchor_type[anchor_type] = query_counts_by_anchor_type.get(anchor_type, 0) + 1

    video_counts_by_mode: Dict[str, int] = {}
    video_counts_by_anchor_type: Dict[str, int] = {}
    provenance_counts_by_mode: Dict[str, int] = {}
    provenance_counts_by_anchor_type: Dict[str, int] = {}

    for record in records:
        prov_list = record.get("retrieval_provenance") or []
        if not isinstance(prov_list, list):
            prov_list = []
        record_modes = set()
        record_anchor_types = set()
        for entry in prov_list:
            if not isinstance(entry, dict):
                continue
            mode = entry.get("retrieval_mode") or "unknown"
            anchor_type = entry.get("anchor_type") or "unknown"
            provenance_counts_by_mode[mode] = provenance_counts_by_mode.get(mode, 0) + 1
            provenance_counts_by_anchor_type[anchor_type] = provenance_counts_by_anchor_type.get(anchor_type, 0) + 1
            record_modes.add(mode)
            record_anchor_types.add(anchor_type)

        for mode in record_modes:
            video_counts_by_mode[mode] = video_counts_by_mode.get(mode, 0) + 1
        for anchor_type in record_anchor_types:
            video_counts_by_anchor_type[anchor_type] = video_counts_by_anchor_type.get(anchor_type, 0) + 1

    return {
        "query_counts_by_mode": query_counts_by_mode,
        "query_counts_by_anchor_type": query_counts_by_anchor_type,
        "video_counts_by_mode": video_counts_by_mode,
        "video_counts_by_anchor_type": video_counts_by_anchor_type,
        "provenance_counts_by_mode": provenance_counts_by_mode,
        "provenance_counts_by_anchor_type": provenance_counts_by_anchor_type,
    }


def write_metadata(records: Iterable[Dict], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.max_queries_per_emotion is not None and args.max_queries_per_emotion < 1:
        raise SystemExit("--max-queries-per-emotion must be >= 1.")
    if args.max_scrolls < 0:
        raise SystemExit("--max-scrolls must be >= 0.")
    if args.min_scroll_novelty < 0 or args.min_scroll_novelty > 1:
        raise SystemExit("--min-scroll-novelty must be between 0 and 1.")
    if args.scroll_novelty_patience < 1:
        raise SystemExit("--scroll-novelty-patience must be >= 1.")
    if args.min_new_videos_per_query < 0:
        raise SystemExit("--min-new-videos-per-query must be >= 0.")
    if args.query_patience < 1:
        raise SystemExit("--query-patience must be >= 1.")

    emotions = resolve_emotions(args.emotions, args.all_emotions)
    output_root: pathlib.Path = args.output_dir
    if args.run_subdir:
        run_id = args.run_id or default_run_id()
        if args.dry_run:
            base_output_dir = output_root / sanitize_for_path(run_id, default="run")
        else:
            base_output_dir = create_unique_run_dir(output_root, run_id)
    else:
        if args.run_id:
            raise SystemExit("--run-id requires --run-subdir.")
        base_output_dir = output_root
        base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output dir: {base_output_dir.resolve()}")

    anchors_cfg = default_anchors_cfg()
    max_queries_cap = args.max_queries_per_emotion

    query_pool = build_queries(
        emotions=emotions,
        anchors_cfg=anchors_cfg,
        max_queries_per_emotion=None,
        seed=args.seed,
    )
    query_specs_by_emotion: Dict[str, List[Dict]] = {emotion: [] for emotion in emotions}
    for spec in query_pool:
        emotion = spec.get("emotion")
        if emotion in query_specs_by_emotion:
            query_specs_by_emotion[emotion].append(spec)

    queries_plan: Dict[str, List[Dict]] = {}
    for emotion in emotions:
        specs = query_specs_by_emotion.get(emotion, [])
        if max_queries_cap is not None:
            specs = specs[:max_queries_cap]
        queries_plan[emotion] = specs

    total_queries = sum(len(specs) for specs in queries_plan.values())
    combo_count = len(emotions)
    print(
        f"[info] Preparing {total_queries} queries across {combo_count} emotion batches on YouTube "
        f"(anchor-types=all, seed={args.seed}, max-queries-cap={max_queries_cap})."
    )
    for emotion in emotions:
        specs = queries_plan.get(emotion, [])
        counts_by_mode: Dict[str, int] = {}
        for spec in specs:
            mode = spec.get("retrieval_mode") or "unknown"
            counts_by_mode[mode] = counts_by_mode.get(mode, 0) + 1
        mode_summary = ", ".join(f"{k}={v}" for k, v in sorted(counts_by_mode.items()))
        print(f"  [{emotion}] {len(specs)} queries ({mode_summary})")
        for spec in specs:
            mode = spec.get("retrieval_mode")
            anchor_type = spec.get("anchor_type")
            print(f"    - [{mode}/{anchor_type}] {spec.get('query')}")

    if args.dry_run:
        print("[info] Dry run; exiting before network calls.")
        return

    if not selenium_available():
        raise SystemExit("selenium is not installed; cannot run searches.")

    emotion_stats: Dict[str, Dict] = {
        emotion: {
            "videos": 0,
            "duplicates_removed": 0,
            "queries_planned": len(queries_plan.get(emotion, [])),
            "queries_executed": 0,
            "query_stop_reason": None,
            "retrieval": {
                "query_counts_by_mode": {},
                "query_counts_by_anchor_type": {},
                "video_counts_by_mode": {},
                "video_counts_by_anchor_type": {},
                "provenance_counts_by_mode": {},
                "provenance_counts_by_anchor_type": {},
            },
        }
        for emotion in emotions
    }

    driver = create_selenium_driver(
        headless=args.headless,
    )
    try:
        for emotion in emotions:
            query_specs = queries_plan.get(emotion, [])
            if not query_specs:
                continue
            records_by_id: Dict[str, Dict] = {}
            duplicates_removed = 0
            executed_query_specs: List[Dict] = []
            query_debug_stats: List[Dict] = []
            low_gain_query_streak = 0
            emotion_stop_reason: Optional[str] = None
            print(f"[info] ===== Emotion: {emotion} =====")
            for query_spec in query_specs:
                query = query_spec.get("query") or ""
                mode = query_spec.get("retrieval_mode")
                anchor_type = query_spec.get("anchor_type")
                print(f"[info] Searching [{mode}/{anchor_type}]: {query}")
                unique_before = len(records_by_id)
                scroll_debug: Dict[str, object] = {}
                try:
                    hits = search_youtube_selenium(
                        driver,
                        query=query,
                        max_results=args.results_per_query,
                        scroll_pause_seconds=args.scroll_pause_seconds,
                        max_scrolls=args.max_scrolls,
                        known_video_ids=records_by_id.keys(),
                        min_scroll_novelty=args.min_scroll_novelty,
                        scroll_novelty_patience=args.scroll_novelty_patience,
                        debug=scroll_debug,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[error] Selenium search failed for '{query}': {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                filtered_hits = filter_results(
                    hits,
                    min_duration_seconds=args.min_duration_seconds,
                    max_duration_seconds=args.max_duration_seconds,
                )
                print(
                    f"[info] Found {len(hits)} videos for query '{query}' "
                    f"({len(filtered_hits)} after filtering)"
                )
                provenance_entry = make_provenance_entry(query_spec)
                new_videos = 0
                merged_duplicates = 0
                for item in filtered_hits:
                    item["emotion"] = emotion
                    video_id = (item.get("video_id") or "").strip()
                    if not video_id:
                        continue
                    inserted = upsert_video_record(records_by_id, item, provenance_entry)
                    if inserted:
                        new_videos += 1
                    else:
                        merged_duplicates += 1
                duplicates_removed += merged_duplicates
                unique_after = len(records_by_id)
                executed_query_specs.append(query_spec)
                query_debug_stats.append(
                    {
                        "query": query,
                        "retrieval_mode": mode,
                        "anchor_type": anchor_type,
                        "hits_total": len(hits),
                        "hits_filtered": len(filtered_hits),
                        "new_videos": new_videos,
                        "merged_duplicates": merged_duplicates,
                        "unique_before": unique_before,
                        "unique_after": unique_after,
                        "scroll": scroll_debug,
                    }
                )

                if args.min_new_videos_per_query and new_videos < args.min_new_videos_per_query:
                    low_gain_query_streak += 1
                else:
                    low_gain_query_streak = 0

                if args.min_new_videos_per_query and low_gain_query_streak >= args.query_patience:
                    emotion_stop_reason = "low_marginal_gain"
                    print(
                        f"[info] Stopping emotion '{emotion}' after {len(executed_query_specs)} queries "
                        f"(last {args.query_patience} queries added < {args.min_new_videos_per_query} new videos)."
                    )
                    break

            combo_results = list(records_by_id.values())
            if duplicates_removed:
                print(
                    f"[info] Merged {duplicates_removed} duplicate video_id hits for emotion '{emotion}' "
                    f"(provenance preserved)."
                )

            emotion_dir = base_output_dir / sanitize_for_path(emotion)
            emotion_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = emotion_dir / "metadata.jsonl"
            retrieval_summary = summarize_retrieval(combo_results, executed_query_specs)
            if emotion_stop_reason is None:
                pool_count = len(query_specs_by_emotion.get(emotion, []))
                planned_count = len(query_specs)
                if max_queries_cap is not None and planned_count < pool_count:
                    emotion_stop_reason = "max_queries_cap"
                else:
                    emotion_stop_reason = "exhausted_queries"

            if not combo_results:
                print(f"[warn] No results for emotion '{emotion}'.")
                metadata_path.write_text("", encoding="utf-8")
                timestamp = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
                summary = {
                    "collected_at": timestamp,
                    "emotion": emotion,
                    "query_count": len(executed_query_specs),
                    "query_plan_count": len(query_specs),
                    "result_count": 0,
                    "output_dir": str(emotion_dir.resolve()),
                    "duplicates_removed": duplicates_removed,
                    "retrieval": retrieval_summary,
                    "adaptive": {
                        "query_stop_reason": emotion_stop_reason,
                        "min_new_videos_per_query": args.min_new_videos_per_query,
                        "query_patience": args.query_patience,
                        "min_scroll_novelty": args.min_scroll_novelty,
                        "scroll_novelty_patience": args.scroll_novelty_patience,
                        "max_scrolls_cap": args.max_scrolls,
                        "query_stats": query_debug_stats,
                    },
                }
                summary_path = emotion_dir / "summary.json"
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                emotion_stats[emotion]["duplicates_removed"] += duplicates_removed
                emotion_stats[emotion]["queries_executed"] += len(executed_query_specs)
                emotion_stats[emotion]["query_stop_reason"] = emotion_stop_reason
                for key, counts in retrieval_summary.items():
                    target = emotion_stats[emotion]["retrieval"].get(key, {})
                    for k, v in counts.items():
                        target[k] = target.get(k, 0) + v
                    emotion_stats[emotion]["retrieval"][key] = target
                continue

            for item in combo_results:
                if "watch_url" not in item and item.get("video_id"):
                    item["watch_url"] = f"https://www.youtube.com/watch?v={item['video_id']}"
                item["download_status"] = "metadata_only"

            timestamp = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            summary = {
                "collected_at": timestamp,
                "emotion": emotion,
                "query_count": len(executed_query_specs),
                "query_plan_count": len(query_specs),
                "result_count": len(combo_results),
                "output_dir": str(emotion_dir.resolve()),
                "duplicates_removed": duplicates_removed,
                "retrieval": retrieval_summary,
                "adaptive": {
                    "query_stop_reason": emotion_stop_reason,
                    "min_new_videos_per_query": args.min_new_videos_per_query,
                    "query_patience": args.query_patience,
                    "min_scroll_novelty": args.min_scroll_novelty,
                    "scroll_novelty_patience": args.scroll_novelty_patience,
                    "max_scrolls_cap": args.max_scrolls,
                    "query_stats": query_debug_stats,
                },
            }
            print(f"[info] Writing metadata for {len(combo_results)} videos to {metadata_path}")
            write_metadata(combo_results, metadata_path)

            summary_path = emotion_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[info] Wrote summary to {summary_path}")
            emotion_stats[emotion]["videos"] += len(combo_results)
            emotion_stats[emotion]["duplicates_removed"] += duplicates_removed
            emotion_stats[emotion]["queries_executed"] += len(executed_query_specs)
            emotion_stats[emotion]["query_stop_reason"] = emotion_stop_reason
            for key, counts in retrieval_summary.items():
                target = emotion_stats[emotion]["retrieval"].get(key, {})
                for k, v in counts.items():
                    target[k] = target.get(k, 0) + v
                emotion_stats[emotion]["retrieval"][key] = target
    finally:
        driver.quit()
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    emotion_summary = {
        "collected_at": timestamp,
        "emotions": emotions,
        "seed": args.seed,
        "max_queries_per_emotion": max_queries_cap,
        "max_queries_cap": max_queries_cap,
        "max_scrolls_cap": args.max_scrolls,
        "min_scroll_novelty": args.min_scroll_novelty,
        "scroll_novelty_patience": args.scroll_novelty_patience,
        "min_new_videos_per_query": args.min_new_videos_per_query,
        "query_patience": args.query_patience,
        "stats": emotion_stats,
    }
    emotion_summary_path = base_output_dir / "emotion_summary.json"
    emotion_summary_path.write_text(json.dumps(emotion_summary, indent=2), encoding="utf-8")
    print(f"[info] Wrote emotion summary to {emotion_summary_path}")
    print("[info] Emotion statistics:")
    for emotion in emotions:
        stats = emotion_stats.get(emotion, {})
        print(
            f"  - {emotion}: {stats.get('videos', 0)} videos "
            f"(duplicates removed: {stats.get('duplicates_removed', 0)})"
        )
    print("[info] Done. Review each emotion folder for metadata.jsonl.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warn] Interrupted by user.", file=sys.stderr)
        sys.exit(1)

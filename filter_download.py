#!/usr/bin/env python3
"""
Filter + download YouTube videos with Qwen2.5-Omni-3B (Hugging Face Transformers).

Install deps (Python):
  - pip install -U "torch" "transformers" "pillow" "soundfile"
  - Optional but often needed by HF models: pip install -U "accelerate" "sentencepiece"
  - Qwen Omni helper: ensure `qwen_omni_utils` (for `process_mm_info`) is importable
    (it is provided in Qwen Omni example code; add it to your PYTHONPATH if needed).

Install external tools (required):
  - yt-dlp  (https://github.com/yt-dlp/yt-dlp)
  - ffmpeg  (includes ffprobe)

Typical installs:
  - pip install -U yt-dlp
  - Ubuntu/Debian: sudo apt-get install -y ffmpeg
  - macOS: brew install yt-dlp ffmpeg

If yt-dlp fails with "Sign in to confirm you’re not a bot":
  - Prefer passing YouTube cookies from a logged-in session:
      --yt_cookies /path/to/cookies.txt
        (Netscape cookies.txt format; browser extensions can export this)
  - On shared/datacenter IPs you may still need a different network / IP.

What this script does
  - Reads JSONL (preferred) or JSON containing YouTube links.
  - For each item: downloads the video, extracts up to 3 short video clips
    (start/middle/end) with ffmpeg, and asks Qwen2.5-Omni-3B
    whether the video contains BOTH:
      (1) a social situation, AND
      (2) at least one target emotion from a user-provided list.
  - If model says DROP (or output is unclear): deletes the video immediately.
  - If model says KEEP: keeps the video as <keep_dir>/<video_id>.mp4 and appends a
    result record to an output JSONL.

Resumable behavior
  - If <out_jsonl> already contains an entry with the same video_id, the item is
    skipped unless --force_redownload is set.

Input format
  --input can be either:
    1) JSONL: each line is a dict with at least {"video_id": "...", "url": "..."}
       (video_id can be missing; url can also be provided as "watch_url".)
    2) JSON: either a list of dicts or {"items":[...]} with the same fields.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import re
import shlex
import shutil
import subprocess
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


# -----------------------------
# Configurable constants
# -----------------------------

NUM_CLIPS = 3
CLIP_DURATION_SEC = 20
USE_AUDIO_IN_VIDEO = True

# Downsample clips before sending to the model to avoid extremely long multimodal token sequences.
FILTER_CLIP_FPS = 1
FILTER_CLIP_MAX_SIDE = 512
FILTER_CLIP_VIDEO_CRF = 28
FILTER_CLIP_VIDEO_PRESET = "veryfast"
FILTER_CLIP_AUDIO_SAMPLE_RATE = 16000
FILTER_CLIP_AUDIO_CHANNELS = 1

YT_DLP_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
MAX_NEW_TOKENS = 256


# -----------------------------
# Small utilities
# -----------------------------


def iso_timestamp_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:n]


def sanitize_id(value: str, default: str = "item") -> str:
    """Make a filesystem-safe id token."""
    base = (value or "").strip()
    if not base:
        return default
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or default


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    timeout_sec: Optional[int] = None,
    check: bool = True,
    log_prefix: str = "",
) -> subprocess.CompletedProcess:
    logging.debug("%s$ %s", log_prefix, " ".join(cmd))
    proc = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if check and proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        msg = "\n".join([part for part in [stdout, stderr] if part])
        raise RuntimeError(msg or f"Command failed with exit code {proc.returncode}: {cmd}")
    return proc


# -----------------------------
# Input / output
# -----------------------------


def iter_jsonl(path: pathlib.Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict in {path} at line {line_no}, got {type(obj).__name__}")
            yield obj


def iter_input_items(path: pathlib.Path) -> Iterator[Dict]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".jsonlines"):
        yield from iter_jsonl(path)
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list or a dict with an 'items' list.")
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"JSON input item #{idx} must be a dict, got {type(item).__name__}")
            yield item
        return

    # Default to JSONL for unknown extensions (preferred).
    yield from iter_jsonl(path)


def extract_youtube_id(url: str) -> str:
    """Best-effort extraction of a YouTube video id from a URL."""
    url = (url or "").strip()
    if not url:
        return ""
    # Common patterns:
    #   https://www.youtube.com/watch?v=VIDEOID
    #   https://youtu.be/VIDEOID
    #   https://www.youtube.com/shorts/VIDEOID
    try:
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        if "youtu.be" in host:
            candidate = path.strip("/").split("/")[0]
            return candidate
        if "youtube.com" in host:
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return (qs["v"][0] or "").strip()
            m = re.match(r"^/(shorts|embed)/([^/?#]+)", path)
            if m:
                return (m.group(2) or "").strip()
    except Exception:  # noqa: BLE001
        return ""
    return ""


def normalize_input_item(item: Dict) -> Tuple[str, str]:
    url = (item.get("url") or item.get("watch_url") or "").strip()
    if not url:
        raise ValueError("Missing required field: url (or watch_url).")

    raw_id = (item.get("video_id") or "").strip()
    if not raw_id:
        raw_id = extract_youtube_id(url) or sha1_short(url)
    video_id = sanitize_id(raw_id, default=sha1_short(url))
    return video_id, url


def load_processed_ids(out_jsonl: pathlib.Path) -> set:
    processed = set()
    if not out_jsonl.exists():
        return processed
    for line_no, line in enumerate(out_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("Skipping invalid JSONL line in %s:%d", out_jsonl, line_no)
            continue
        if isinstance(obj, dict):
            vid = (obj.get("video_id") or "").strip()
            if vid:
                processed.add(vid)
    return processed


def append_jsonl(path: pathlib.Path, record: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# -----------------------------
# Downloading
# -----------------------------


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".flv"}


def find_existing_video(work_dir: pathlib.Path, video_id: str) -> Optional[pathlib.Path]:
    candidates = []
    for path in work_dir.glob(video_id + ".*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def download_video(
    *,
    url: str,
    video_id: str,
    work_dir: pathlib.Path,
    force_redownload: bool,
    cookies_path: Optional[pathlib.Path] = None,
    extra_yt_dlp_args: Sequence[str] = (),
) -> pathlib.Path:
    ensure_dir(work_dir)
    existing = find_existing_video(work_dir, video_id)
    if existing and not force_redownload:
        logging.info("Reusing existing download: %s", existing)
        return existing

    outtmpl = str(work_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
        "-f",
        YT_DLP_FORMAT,
        "--merge-output-format",
        "mp4",
        "-o",
        outtmpl,
        "--print",
        "after_move:filepath",
    ]
    if cookies_path is not None:
        cmd += ["--cookies", str(cookies_path)]
    if extra_yt_dlp_args:
        cmd += list(extra_yt_dlp_args)
    cmd.append("--force-overwrites" if force_redownload else "--no-overwrites")
    cmd.append(url)

    try:
        proc = run_cmd(cmd, check=True, log_prefix="[yt-dlp] ")
    except RuntimeError as exc:
        msg = str(exc)
        if "confirm you're not a bot" in msg.lower():
            raise RuntimeError(
                msg
                + "\n\nYouTube is requiring human verification. Typical fixes:\n"
                + "  - Update yt-dlp to the latest version.\n"
                + "  - Pass cookies from a logged-in session via --yt_cookies.\n"
                + "  - If you're on a shared/datacenter IP, try a different network / IP.\n"
            ) from exc
        raise
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    for candidate in reversed(lines):
        path = pathlib.Path(candidate)
        if path.exists() and path.is_file():
            return path

    # If yt-dlp didn't print the final path (can happen with --no-overwrites), locate it.
    existing = find_existing_video(work_dir, video_id)
    if existing:
        return existing
    raise RuntimeError("yt-dlp finished but the downloaded file could not be located.")


# -----------------------------
# ffmpeg sampling
# -----------------------------


def probe_duration_sec(video_path: pathlib.Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = run_cmd(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
            )
            value = (proc.stdout or "").strip()
            if not value or value.upper() == "N/A":
                raise ValueError(f"ffprobe returned non-numeric duration: {value!r}")
            duration = float(value)
            if duration <= 0:
                raise ValueError(f"ffprobe returned non-positive duration: {duration}")
            return duration
        except Exception as exc:  # noqa: BLE001
            logging.debug("ffprobe duration probe failed (%s): %s", video_path, exc)

    # Fallback: parse ffmpeg stderr.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        proc = run_cmd([ffmpeg, "-hide_banner", "-i", str(video_path)], check=False)
        text = (proc.stderr or "") + "\n" + (proc.stdout or "")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if not m:
            return None
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = float(m.group(3))
        duration = hours * 3600 + minutes * 60 + seconds
        return duration if duration > 0 else None
    except Exception as exc:  # noqa: BLE001
        logging.debug("ffmpeg duration probe failed (%s): %s", video_path, exc)
        return None


def compute_clip_starts(duration_sec: Optional[float]) -> List[float]:
    if not duration_sec or duration_sec <= 0:
        return [0.0]
    if duration_sec <= CLIP_DURATION_SEC:
        return [0.0]
    starts = [
        0.0,
        max(duration_sec / 2.0 - CLIP_DURATION_SEC / 2.0, 0.0),
        max(duration_sec - CLIP_DURATION_SEC, 0.0),
    ]
    # De-duplicate near-identical starts for short clips.
    unique: List[float] = []
    for s in starts[:NUM_CLIPS]:
        if not any(abs(s - u) < 1.0 for u in unique):
            unique.append(s)
    return unique


def extract_video_clip(
    *,
    video_path: pathlib.Path,
    out_path: pathlib.Path,
    start_sec: float,
    duration_sec: int,
) -> pathlib.Path:
    ensure_dir(out_path.parent)
    # Always re-encode to a low-fps, low-res clip so the model input stays within max length.
    vf = f"fps={FILTER_CLIP_FPS},scale='min({FILTER_CLIP_MAX_SIDE},iw)':-2"
    cmd_x264 = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        str(duration_sec),
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        FILTER_CLIP_VIDEO_PRESET,
        "-crf",
        str(FILTER_CLIP_VIDEO_CRF),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ac",
        str(FILTER_CLIP_AUDIO_CHANNELS),
        "-ar",
        str(FILTER_CLIP_AUDIO_SAMPLE_RATE),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        run_cmd(cmd_x264, check=True, log_prefix="[ffmpeg] ")
    except Exception as exc:  # noqa: BLE001
        # Fallback: use mpeg4 encoder if libx264 is unavailable.
        logging.warning("ffmpeg/libx264 failed; falling back to mpeg4 (%s): %s", out_path, exc)
        cmd_mpeg4 = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            str(duration_sec),
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-sn",
            "-vf",
            vf,
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            str(FILTER_CLIP_AUDIO_CHANNELS),
            "-ar",
            str(FILTER_CLIP_AUDIO_SAMPLE_RATE),
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        run_cmd(cmd_mpeg4, check=True, log_prefix="[ffmpeg] ")
    if not out_path.exists():
        raise RuntimeError(f"Failed to extract clip: {out_path}")
    return out_path


def extract_video_clips(
    *,
    video_path: pathlib.Path,
    clips_dir: pathlib.Path,
) -> List[pathlib.Path]:
    ensure_dir(clips_dir)
    duration = probe_duration_sec(video_path)
    starts = compute_clip_starts(duration)
    clips: List[pathlib.Path] = []
    for idx, start in enumerate(starts[:NUM_CLIPS]):
        clip_path = clips_dir / f"clip_{idx}.mp4"
        try:
            clips.append(
                extract_video_clip(
                    video_path=video_path,
                    out_path=clip_path,
                    start_sec=start,
                    duration_sec=CLIP_DURATION_SEC,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Clip extraction failed (%s): %s", video_path, exc)
    if not clips:
        raise RuntimeError("Failed to extract any video clips.")
    return clips


# -----------------------------
# Model (Transformers)
# -----------------------------


SYSTEM_PROMPT = """You are a strict dataset curator for social emotions.

You will receive a small sample of a video: up to 3 short clips from the start/middle/end (with audio).

Decide whether to KEEP or DROP the video for a social-emotion dataset with the following rules:
- Return STRICT JSON ONLY (no markdown, no extra text, no code fences).
- If evidence is unclear, choose DROP.
- "Social situation" means: at least two people interacting OR a clear social context (argument, apology, celebration,
  embarrassment, comfort, rejection, conflict, negotiation, persuasion, public recognition, etc.).
- "Target emotion" must match any label in the provided target emotion list.
"""


def build_user_prompt(
    *,
    video_id: str,
    url: str,
    target_emotions: Sequence[str],
) -> str:
    emotions = [e.strip() for e in target_emotions if e.strip()]
    emotions_str = ", ".join(emotions) if emotions else "(none)"
    return (
        f"Video ID: {video_id}\n"
        f"URL: {url}\n"
        f"Target emotions: {emotions_str}\n\n"
        "Return strict JSON ONLY with this schema:\n"
        "{\n"
        '  "decision": "KEEP" or "DROP",\n'
        '  "has_social_situation": true/false,\n'
        '  "has_target_emotion": true/false,\n'
        '  "target_emotions_found": ["..."],\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reason": "one sentence"\n'
        "}\n"
    )


def extract_first_json_object(text: str) -> Optional[Dict]:
    """Extract the first JSON object from text (robust to leading/trailing chatter)."""
    s = (text or "").strip()
    if not s:
        return None

    start = s.find("{")
    if start == -1:
        return None

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = s[start : i + 1]
                try:
                    obj = json.loads(snippet)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def validate_model_output(obj: Dict, target_emotions: Sequence[str]) -> Dict:
    """Validate + normalize the model JSON; if invalid, return a DROP decision."""
    fallback = {
        "decision": "DROP",
        "has_social_situation": False,
        "has_target_emotion": False,
        "target_emotions_found": [],
        "confidence": 0.0,
        "reason": "invalid_or_unclear_model_output",
    }
    if not isinstance(obj, dict):
        return fallback

    decision = str(obj.get("decision") or "").strip().upper()
    if decision not in ("KEEP", "DROP"):
        return fallback

    has_social = obj.get("has_social_situation")
    has_emotion = obj.get("has_target_emotion")
    if not isinstance(has_social, bool) or not isinstance(has_emotion, bool):
        return fallback

    found = obj.get("target_emotions_found")
    if not isinstance(found, list) or not all(isinstance(x, str) for x in found):
        return fallback

    try:
        confidence = float(obj.get("confidence"))
    except Exception:  # noqa: BLE001
        return fallback
    if confidence < 0.0 or confidence > 1.0:
        return fallback

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return fallback

    normalized_targets = {normalize_label(x) for x in target_emotions if normalize_label(x)}
    normalized_found = [normalize_label(x) for x in found if normalize_label(x)]
    has_match = bool(normalized_targets.intersection(normalized_found))

    # Enforce the task definition strictly: KEEP only if both conditions are true and there is a label match.
    if decision == "KEEP" and (not has_social or not has_emotion or not has_match):
        return {
            **fallback,
            "reason": "inconsistent_keep_without_required_evidence",
        }
    if has_emotion and not has_match:
        return {
            **fallback,
            "reason": "emotion_not_in_target_list",
        }

    return {
        "decision": decision,
        "has_social_situation": has_social,
        "has_target_emotion": has_emotion,
        "target_emotions_found": found,
        "confidence": confidence,
        "reason": reason.strip(),
    }


class QwenOmniRunner:
    """Loads Qwen2.5-Omni-3B once and runs a single gating prompt."""

    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._process_mm_info = None

    def ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore
            from qwen_omni_utils import process_mm_info  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Missing Python deps. Install: pip install -U torch transformers pillow soundfile "
                "(and ensure qwen_omni_utils is available in your env)."
            ) from exc

        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        self._process_mm_info = process_mm_info

        if self.device == "cuda":
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto"
            )
        else:
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto"
            )
            self._model.to("cpu")
        self._model.eval()

    def generate_gate_json(
        self,
        *,
        video_id: str,
        url: str,
        target_emotions: Sequence[str],
        video_paths: Sequence[pathlib.Path],
    ) -> Tuple[Optional[Dict], str]:
        self.ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        assert self._process_mm_info is not None

        user_prompt = build_user_prompt(video_id=video_id, url=url, target_emotions=target_emotions)

        def infer_max_input_tokens() -> int:
            config = getattr(self._model, "config", None)
            for attr in (
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
                "model_max_length",
            ):
                value = getattr(config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
            text_cfg = getattr(config, "text_config", None)
            value = getattr(text_cfg, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                return value
            return 32768

        max_len = infer_max_input_tokens()
        candidate_sets: List[Sequence[pathlib.Path]] = [list(video_paths)]
        if len(video_paths) > 1:
            candidate_sets.append([video_paths[0]])

        out = None
        raw_text = ""
        for candidate_videos in candidate_sets:
            video_items = [{"type": "video", "video": str(p)} for p in candidate_videos]
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": ([{"type": "text", "text": user_prompt}] + video_items)},
            ]

            text = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = self._process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = self._processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
            inputs = inputs.to(self._model.device).to(self._model.dtype)

            input_ids = inputs.get("input_ids")
            input_len = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else 0  # type: ignore[union-attr]
            if input_len >= max_len:
                logging.warning(
                    "Model input too long (%d >= %d) with %d clip(s); trying fewer clips.",
                    input_len,
                    max_len,
                    len(candidate_videos),
                )
                continue

            max_new = min(MAX_NEW_TOKENS, max_len - input_len - 1)
            if max_new < 1:
                logging.warning(
                    "No room for generation (input_len=%d, max_len=%d); trying fewer clips.",
                    input_len,
                    max_len,
                )
                continue

            tokenizer = getattr(self._processor, "tokenizer", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)

            out = self._model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            if isinstance(out, (tuple, list)):
                text_ids = out[0]
            else:
                text_ids = out

            # Decode only the generated continuation (exclude the prompt) so logs/results don't include input.
            if hasattr(input_ids, "shape") and hasattr(text_ids, "shape"):
                trimmed = text_ids[:, input_ids.shape[-1] :]  # type: ignore[union-attr]
            else:
                trimmed = text_ids

            decoded = self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_text = decoded[0] if isinstance(decoded, list) and decoded else str(decoded)
            break

        if out is None and not raw_text:
            return None, "MODEL_INPUT_TOO_LONG"

        obj = extract_first_json_object(raw_text)
        return (obj if isinstance(obj, dict) else None), raw_text



# -----------------------------
# Keep / drop file handling
# -----------------------------


def ensure_mp4_in_keep_dir(
    *,
    video_path: pathlib.Path,
    keep_dir: pathlib.Path,
    video_id: str,
    force: bool,
) -> pathlib.Path:
    ensure_dir(keep_dir)
    target = keep_dir / f"{video_id}.mp4"
    if target.exists() and force:
        target.unlink()

    if video_path.suffix.lower() == ".mp4":
        if target.exists():
            raise RuntimeError(f"Target already exists: {target} (use --force_redownload)")
        shutil.move(str(video_path), str(target))
        return target

    # Remux/convert to mp4 (best-effort).
    tmp_target = target.with_suffix(".mp4.tmp")
    if tmp_target.exists():
        tmp_target.unlink()
    try:
        run_cmd(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(tmp_target),
            ],
            check=True,
            log_prefix="[ffmpeg] ",
        )
    except Exception:
        # Fallback: re-encode (may be slower; requires encoder availability).
        run_cmd(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                str(tmp_target),
            ],
            check=True,
            log_prefix="[ffmpeg] ",
        )

    if not tmp_target.exists():
        raise RuntimeError("Failed to produce an mp4 output.")
    if target.exists():
        target.unlink()
    tmp_target.replace(target)
    video_path.unlink(missing_ok=True)
    return target


# -----------------------------
# Main processing loop
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download videos and filter them with Qwen2.5-Omni-3B.")
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Input JSONL/JSON path.")
    parser.add_argument("--out_jsonl", type=pathlib.Path, default=pathlib.Path("results.jsonl"))
    parser.add_argument("--keep_dir", type=pathlib.Path, default=pathlib.Path("kept_videos"))
    parser.add_argument("--tmp_dir", type=pathlib.Path, default=pathlib.Path("tmp_downloads"))
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--target_emotions", type=str, default="embarrassment")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument("--min_duration_sec", type=int, default=0)
    parser.add_argument("--max_duration_sec", type=int, default=180)
    parser.add_argument("--force_redownload", action="store_true")
    parser.add_argument(
        "--yt_cookies",
        type=pathlib.Path,
        default=None,
        help="Path to a Netscape cookies.txt file for yt-dlp (recommended to avoid bot checks).",
    )
    parser.add_argument(
        "--yt_dlp_args",
        type=str,
        default="",
        help='Extra yt-dlp args (shell-like string), e.g. "--sleep-requests --sleep-interval 1".',
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"--input not found: {args.input}")

    # Convenience: if no cookie option is provided, auto-detect a cookies file
    # next to this script (common export filenames).
    if args.yt_cookies is None:
        script_dir = pathlib.Path(__file__).resolve().parent
        for name in ("cookies.txt", "www.youtube.com_cookies.txt"):
            candidate = script_dir / name
            if candidate.exists():
                args.yt_cookies = candidate
                logging.info("Auto-using yt-dlp cookies file: %s", candidate)
                break

    if args.yt_cookies is not None and not args.yt_cookies.exists():
        raise SystemExit(f"--yt_cookies not found: {args.yt_cookies}")
    if args.min_duration_sec is not None and args.min_duration_sec < 0:
        raise SystemExit("--min_duration_sec must be >= 0")
    if args.max_duration_sec is not None and args.max_duration_sec < 0:
        raise SystemExit("--max_duration_sec must be >= 0")
    if (
        args.min_duration_sec is not None
        and args.max_duration_sec is not None
        and args.min_duration_sec > args.max_duration_sec
    ):
        raise SystemExit("--min_duration_sec must be <= --max_duration_sec")

    target_emotions = [x.strip() for x in (args.target_emotions or "").split(",") if x.strip()]
    if not target_emotions:
        raise SystemExit("--target_emotions cannot be empty")

    processed_ids = load_processed_ids(args.out_jsonl)
    logging.info("Already processed: %d", len(processed_ids))

    runner = QwenOmniRunner(model_id=args.model_id, device=args.device)
    try:
        extra_yt_dlp_args = shlex.split((args.yt_dlp_args or "").strip())
    except ValueError as exc:
        raise SystemExit(f"Invalid --yt_dlp_args: {exc}") from exc

    processed_now = 0
    for raw_item in iter_input_items(args.input):
        try:
            video_id, url = normalize_input_item(raw_item)
        except Exception as exc:  # noqa: BLE001
            logging.error("Skipping invalid input item: %s", exc)
            continue

        if (not args.force_redownload) and video_id in processed_ids:
            logging.info("Skipping already processed video_id=%s", video_id)
            continue

        ts = iso_timestamp_utc()
        work_dir = args.tmp_dir / video_id
        samples_dir = work_dir / "samples"

        record: Dict = {
            "video_id": video_id,
            "url": url,
            "downloaded_path": None,
            "duration_sec": None,
            "decision": None,
            "model_output_json": None,
            "model_output_raw": None,
            "error": None,
            "timestamp": ts,
        }

        try:
            if work_dir.exists() and args.force_redownload:
                shutil.rmtree(work_dir, ignore_errors=True)
            ensure_dir(work_dir)

            logging.info("Downloading video_id=%s url=%s", video_id, url)
            downloaded = download_video(
                url=url,
                video_id=video_id,
                work_dir=work_dir,
                force_redownload=args.force_redownload,
                cookies_path=args.yt_cookies,
                extra_yt_dlp_args=extra_yt_dlp_args,
            )

            # Duration gate (optional).
            duration = probe_duration_sec(downloaded)
            record["duration_sec"] = duration
            if duration is not None:
                if args.min_duration_sec is not None and duration < args.min_duration_sec:
                    model_obj = {
                        "decision": "DROP",
                        "has_social_situation": False,
                        "has_target_emotion": False,
                        "target_emotions_found": [],
                        "confidence": 1.0,
                        "reason": f"duration_below_min_{args.min_duration_sec}",
                    }
                    record["decision"] = "DROP"
                    record["model_output_json"] = model_obj
                    downloaded.unlink(missing_ok=True)
                    append_jsonl(args.out_jsonl, record)
                    processed_ids.add(video_id)
                    processed_now += 1
                    continue
                if args.max_duration_sec is not None and duration > args.max_duration_sec:
                    model_obj = {
                        "decision": "DROP",
                        "has_social_situation": False,
                        "has_target_emotion": False,
                        "target_emotions_found": [],
                        "confidence": 1.0,
                        "reason": f"duration_above_max_{args.max_duration_sec}",
                    }
                    record["decision"] = "DROP"
                    record["model_output_json"] = model_obj
                    downloaded.unlink(missing_ok=True)
                    append_jsonl(args.out_jsonl, record)
                    processed_ids.add(video_id)
                    processed_now += 1
                    continue

            # Extract short clips for faster filtering.
            clips_dir = samples_dir / "clips"
            try:
                clip_paths = extract_video_clips(video_path=downloaded, clips_dir=clips_dir)
                logging.info("Extracted clips video_id=%s clips=%d", video_id, len(clip_paths))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Clip extraction failed; using full video for filtering: %s", exc)
                clip_paths = [downloaded]

            # Model gating.
            model_raw_obj, model_raw_text = runner.generate_gate_json(
                video_id=video_id,
                url=url,
                target_emotions=target_emotions,
                video_paths=clip_paths,
            )
            record["model_output_raw"] = model_raw_text
            validated = validate_model_output(model_raw_obj or {}, target_emotions)
            record["model_output_json"] = validated
            record["decision"] = validated["decision"]

            if validated["decision"] == "KEEP":
                kept_path = ensure_mp4_in_keep_dir(
                    video_path=downloaded,
                    keep_dir=args.keep_dir,
                    video_id=video_id,
                    force=args.force_redownload,
                )
                record["downloaded_path"] = str(kept_path)
                logging.info("KEPT video_id=%s -> %s", video_id, kept_path)
            else:
                # DROP: delete downloaded file immediately.
                downloaded.unlink(missing_ok=True)
                logging.info("DROPPED video_id=%s", video_id)

            append_jsonl(args.out_jsonl, record)
            processed_ids.add(video_id)
            processed_now += 1
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)
            logging.exception("Error processing video_id=%s: %s", video_id, exc)

            # Ensure downloaded file is not kept on disk in error cases.
            try:
                existing = find_existing_video(work_dir, video_id)
                if existing:
                    existing.unlink(missing_ok=True)
            except Exception:
                pass

            append_jsonl(args.out_jsonl, record)
            processed_ids.add(video_id)
            processed_now += 1
        finally:
            # Always clean temp samples and leftovers to save disk.
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    logging.info("Done. Newly processed: %d", processed_now)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")

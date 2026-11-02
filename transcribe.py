import argparse
import os
import re
import sys
import time
from pathlib import Path


def add_cuda_dlls():
    """Windows: make pip-installed cuBLAS/cuDNN DLLs visible to ctranslate2."""
    if sys.platform != "win32":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for bin_dir in nvidia_root.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


add_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac", ".mov", ".aac"}


def fmt_ts(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def is_hallucination(words, low_p, max_ratio=0.5):
    """Most words very unsure -> likely invented text (e.g. 'Thank you for your permission')."""
    if not words:
        return True
    unsure = sum(1 for w in words if w.probability < low_p)
    return unsure / len(words) > max_ratio


def review_text(words, flag_p):
    """Wrap unsure words as [word?] so a teacher can re-listen to just those spots."""
    out = []
    for w in words:
        word = w.word.strip()
        out.append(f"[{word}?]" if w.probability < flag_p else word)
    return " ".join(out)


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def copies_prompt(text, prompt, min_words=5, ratio=0.8):
    """Segment is mostly words from the hint list -> the model echoed the hint."""
    seg_words = _words(text)
    if len(seg_words) < min_words:
        return False
    hint = set(_words(prompt))
    return sum(w in hint for w in seg_words) / len(seg_words) >= ratio


def collect_files(target):
    p = Path(target)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    return [p]


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files with faster-whisper")
    parser.add_argument("input", help="Audio/video file or a folder of them")
    parser.add_argument("--out", default="outputs", help="Folder for .txt/.srt results")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--prompt", default=None, help="Hint text: script excerpt or key words")
    parser.add_argument("--no-vad", action="store_true", help="Disable silence filtering")
    parser.add_argument("--flag-p", type=float, default=0.5,
                        help="Words below this confidence are marked [word?] in .review.txt")
    parser.add_argument("--drop-p", type=float, default=0.3,
                        help="Drop a segment when most of its words are below this confidence")
    args = parser.parse_args()

    files = collect_files(args.input)
    if not files:
        sys.exit(f"No audio/video files found in {args.input}")

    compute_type = "float16" if args.device == "cuda" else "int8"
    print(f"Loading {args.model} on {args.device} ({compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"\n=== {f.name} ===")
        start = time.time()
        common = dict(
            language=args.lang,
            beam_size=args.beam,
            condition_on_previous_text=False,  # reduces repeated/hallucinated text
            temperature=0.0,  # same input -> same output
            word_timestamps=True,
        )
        segments, info = model.transcribe(
            str(f),
            vad_filter=not args.no_vad,
            # hotwords is applied to every 30s window; initial_prompt only reaches the first one
            # when condition_on_previous_text=False
            hotwords=args.prompt,
            **common,
        )

        final = []
        for seg in segments:
            if args.prompt and copies_prompt(seg.text, args.prompt):
                # The model echoed the hint list instead of the child's speech:
                # redo this range without hints, starting where the last good segment ended
                # so speech just before the echo isn't lost.
                redo_start = final[-1].end if final else 0.0
                print(f"[{redo_start:6.2f}s -> {seg.end:6.2f}s] (hint echoed, retrying without hints)")
                redo, _ = model.transcribe(
                    str(f), vad_filter=False, clip_timestamps=[redo_start, seg.end], **common
                )
                final.extend(redo)
            else:
                final.append(seg)

        lines, review, srt, dropped = [], [], [], []
        for seg in final:
            text = seg.text.strip()
            if is_hallucination(seg.words, args.drop_p):
                dropped.append(f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {text}")
                print(f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] (dropped, unsure) {text}")
                continue
            marked = review_text(seg.words, args.flag_p)
            print(f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {marked}")
            lines.append(text)
            review.append(f"[{seg.start:6.2f}s] {marked}")
            srt.append(f"{len(srt) + 1}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n")

        if dropped:
            review += ["", "# Dropped as likely invented text:"] + dropped

        (out_dir / f"{f.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / f"{f.stem}.review.txt").write_text("\n".join(review) + "\n", encoding="utf-8")
        (out_dir / f"{f.stem}.srt").write_text("\n".join(srt), encoding="utf-8")
        took = time.time() - start
        print(f"-> {info.duration:.1f}s audio in {took:.1f}s, saved to {out_dir / f.stem}.txt/.review.txt/.srt")


if __name__ == "__main__":
    main()

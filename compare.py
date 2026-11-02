"""Compare a child's reading against the script.

1. Transcribe without hints (what the child actually said).
2. Align transcript words to script words: match / missed / different / extra.
3. For every mismatch, ask the model how well the audio fits the script wording vs.
   the transcribed wording. If the script wording fits about as well, the model
   probably mis-heard and the child read it right.
"""
import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

import transcribe  # noqa: F401  (sets up CUDA DLLs on Windows)
from transcribe import collect_files, is_hallucination
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer

SR = 16000
NUMBERS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "20": "twenty",
}


def norm(word):
    w = re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))
    return NUMBERS.get(w, w)


def script_words(text):
    words = []
    for raw in text.split():
        n = norm(raw)
        if n:
            words.append({"text": raw, "norm": n})
    return words


def transcribe_words(model, audio, args):
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
        word_timestamps=True,
    )
    words = []
    for seg in segments:
        if is_hallucination(seg.words, args.drop_p):
            continue
        for w in seg.words:
            n = norm(w.word)
            if n:
                words.append({"text": w.word.strip(), "norm": n, "start": w.start, "end": w.end})
    return words


class Scorer:
    """Average log-probability of a given text for an audio clip (teacher-forced)."""

    def __init__(self, model):
        self.model = model
        self.tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual,
                             task="transcribe", language="en")

    def evidence(self, clip, text):
        """How much the audio supports the text beyond plain language fluency:
        score on the clip minus score on silence of the same length."""
        s = self.score(clip, text)
        prior = self.score(np.zeros_like(clip), text)
        if s is None or prior is None:
            return None
        return s - prior

    def score(self, clip, text):
        tokens = self.tok.encode(" " + text.strip())
        if not tokens:
            return None
        feats = self.model.feature_extractor(clip)
        n_frames = min(feats.shape[-1], self.model.feature_extractor.nb_max_frames)
        enc = self.model.encode(pad_or_trim(feats))
        res = self.model.model.align(enc, self.tok.sot_sequence, [tokens], n_frames,
                                     median_filter_width=7)
        probs = np.clip(np.array(res[0].text_token_probs[: len(tokens)]), 1e-6, 1.0)
        return float(np.mean(np.log(probs)))


def compare(ref, hyp):
    """Yield (op, ref_slice, hyp_slice) word-level diff."""
    sm = SequenceMatcher(None, [w["norm"] for w in ref], [w["norm"] for w in hyp], autojunk=False)
    ops = [list(o) for o in sm.get_opcodes()]

    # Merge mismatches split by a single coincidentally matching word
    # ("the end OF the decade" vs "a little bit, a small group OF").
    merged = []
    for o in ops:
        if (merged and o[0] != "equal" and len(merged) >= 2 and merged[-1][0] == "equal"
                and merged[-1][2] - merged[-1][1] == 1 and merged[-2][0] != "equal"):
            eq = merged.pop()
            prev = merged.pop()
            o = ["replace", prev[1], o[2], prev[3], o[4]]
            del eq
        merged.append(o)

    for op, i1, i2, j1, j2 in merged:
        if op != "equal" and "".join(w["norm"] for w in ref[i1:i2]) == "".join(
            w["norm"] for w in hyp[j1:j2]
        ):
            op = "equal"  # e.g. "cyberattacks" vs "cyber attacks"
        if op != "equal" and i2 > i1 and j2 > j1:
            op = "replace"
        yield op, (i1, i2), (j1, j2)


def phrase(words, a, b):
    return " ".join(w["text"] for w in words[a:b])


def check_mismatch(scorer, audio, ref, hyp, rs, hs):
    """Score script wording vs transcribed wording on the same audio, with 1 word of context."""
    (i1, i2), (j1, j2) = rs, hs
    ci1, ci2 = max(i1 - 1, 0), min(i2 + 1, len(ref))
    cj1, cj2 = max(j1 - 1, 0), min(j2 + 1, len(hyp))
    if cj1 >= cj2:
        return None
    start = max(hyp[cj1]["start"] - 0.3, 0)
    end = hyp[cj2 - 1]["end"] + 0.3
    if end - start > 29:
        return None
    clip = audio[int(start * SR): int(end * SR)]
    ref_score = scorer.evidence(clip, phrase(ref, ci1, ci2))
    hyp_score = scorer.evidence(clip, phrase(hyp, cj1, cj2))
    if ref_score is None or hyp_score is None:
        return None
    # Third candidate: the script words were not (clearly) said at all.
    # The model tends to fill a mumbled spot with a plausible word ("minutes").
    omit_score = None
    if i2 > i1:
        context = ref[ci1:i1] + ref[i2:ci2]
        if context:
            omit_score = scorer.evidence(clip, " ".join(w["text"] for w in context))
    return {"start": start, "end": end, "ref_score": ref_score, "hyp_score": hyp_score,
            "omit_score": omit_score, "gap": ref_score - hyp_score}


def main():
    p = argparse.ArgumentParser(description="Compare children's reading against a script")
    p.add_argument("input", help="Audio/video file or folder")
    p.add_argument("--script", required=True, help="Script text file")
    p.add_argument("--out", default="reports")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--drop-p", type=float, default=0.3)
    p.add_argument("--pass-gap", type=float, default=-1.0,
                   help="Script wording within this log-prob of the transcript -> model mis-heard")
    args = p.parse_args()

    ref = script_words(Path(args.script).read_text(encoding="utf-8"))
    files = collect_files(args.input)
    if not files:
        sys.exit(f"No audio/video files found in {args.input}")

    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    scorer = Scorer(model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for f in files:
        audio = decode_audio(str(f), SR)
        hyp = transcribe_words(model, audio, args)

        matched = 0
        rows = []
        for op, rs, hs in compare(ref, hyp):
            n_ref = rs[1] - rs[0]
            if op == "equal":
                matched += n_ref
                continue
            said = phrase(hyp, *hs)
            script = phrase(ref, *rs)
            t = hyp[hs[0]]["start"] if hs[0] < len(hyp) else (hyp[-1]["end"] if hyp else 0)
            if op == "insert" and rs[0] in (0, len(ref)):
                # Greeting / closing words before or after the script: not a reading error.
                rows.append({"t": t, "kind": "대본 밖 발화", "script": "", "said": said,
                             "verdict": "참고", "check": None})
                continue
            check = check_mismatch(scorer, audio, ref, hyp, rs, hs)
            kind = {"replace": "다르게 읽음", "delete": "빠뜨림", "insert": "추가로 말함"}[op]
            omit = check["omit_score"] if check else None
            if (op == "replace" and omit is not None
                    and omit >= max(check["ref_score"], check["hyp_score"])):
                # Saying nothing fits the audio best: mumbled or skipped.
                kind = "불분명/빠뜨림"
                verdict = "확인 필요"
            elif check and check["gap"] >= args.pass_gap:
                verdict = "통과 (모델이 잘못 들음)"
                if op != "insert":
                    matched += n_ref
            else:
                verdict = "확인 필요"
            rows.append({"t": t, "kind": kind, "script": script, "said": said,
                         "verdict": verdict, "check": check})

        rate = matched / len(ref) * 100 if ref else 0
        summary.append((f.stem, rate, sum(r["verdict"] == "확인 필요" for r in rows)))

        lines = [f"# {f.stem}", "", f"대본 일치율: **{rate:.0f}%** ({matched}/{len(ref)} 단어)", "",
                 "| 시간 | 유형 | 대본 | 아이가 읽은 것(받아쓰기) | 판정 | 점수 차 |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            gap = f"{r['check']['gap']:+.2f}" if r["check"] else "-"
            lines.append(f"| {r['t']:.1f}s | {r['kind']} | {r['script'] or '-'} | {r['said'] or '-'} "
                         f"| {r['verdict']} | {gap} |")
        lines += ["", "## 받아쓰기 전문", "", " ".join(w["text"] for w in hyp), ""]
        report = out_dir / f"{f.stem}.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines[:4 + 3 + len(rows)]))
        print(f"-> {report}\n")

    print("=== 요약 ===")
    for name, rate, todo in summary:
        print(f"{name}: 일치율 {rate:.0f}%, 확인 필요 {todo}곳")


if __name__ == "__main__":
    main()

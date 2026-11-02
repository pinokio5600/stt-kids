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
from transcribe import boost_quiet, collect_files, is_hallucination, load_audio, undo_loops
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
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


def script_words(text, extra_names=()):
    words = []
    for line_no, line in enumerate(text.splitlines()):
        for raw in line.split():
            n = norm(raw)
            if n:
                words.append({"text": raw, "norm": n, "line": line_no})

    # Names (Philoctetes, Pegasus, Dario...) are often mis-heard even when read well,
    # so they are left out of the accuracy score. A capitalised word in mid-sentence is a name;
    # every other occurrence of it counts too.
    names = {norm(x) for x in extra_names}
    for prev, w in zip(words, words[1:]):
        if (w["text"][0].isupper() and w["norm"] not in {"i", "i'm", "i'll", "i've", "i'd"}
                and prev["line"] == w["line"] and not prev["text"].endswith((".", "!", "?", ":"))):
            names.add(w["norm"])
    for w in words:
        w["name"] = w["norm"] in names
    return words


def transcribe_words(model, raw, args):
    opts = dict(language="en", beam_size=5, condition_on_previous_text=False,
                temperature=0.0, word_timestamps=True)
    if args.no_boost:
        segments, _ = model.transcribe(raw, vad_filter=True, **opts)
    else:
        segments, _ = model.transcribe(boost_quiet(raw), vad_filter=True, **opts)
        segments = undo_loops(model, segments, raw, **opts)
    words = []
    for seg in segments:
        if is_hallucination(seg.words, args.drop_p):
            continue
        for w in seg.words:
            n = norm(w.word)
            if n:
                words.append({"text": w.word.strip(), "norm": n, "start": w.start, "end": w.end,
                              "p": w.probability})
    return words


# Score weights and the ranges that map a measurement to 0..1 of its points.
WEIGHTS = {"accuracy": 60, "fluency": 25, "clarity": 15}
WPM_RANGE = (80, 160)      # words per minute inside a sentence: 80 or less -> 0, 160 or more -> full
PAUSE_SEC = 1.5            # a stop this long inside a sentence counts as hesitation
PAUSE_COST = 0.2           # each hesitation takes 20% of the pause half of fluency
CLARITY_RANGE = (0.8, 0.96)  # average model confidence on correctly read words
GRADES = [(85, "A"), (70, "B"), (0, "C")]  # provisional: tune against teacher grades
MIN_HEARD = 0.3            # heard fewer words than 30% of the script -> can't score


def scale(x, lo, hi):
    return min(max((x - lo) / (hi - lo), 0.0), 1.0)


def score_reading(ref, hyp, correct, pairs):
    """pairs: (ref index, hyp index) for words read exactly as written."""
    if len(hyp) < MIN_HEARD * len(ref):
        return None

    content = [i for i, w in enumerate(ref) if not w["name"]]
    accuracy = sum(correct[i] for i in content) / len(content) if content else 0

    # Fluency is measured within each script line only: the gap between lines
    # may just be waiting for the next character's turn in the video.
    by_line = {}
    for i, j in pairs:
        by_line.setdefault(ref[i]["line"], []).append((i, j))
    words = span = pauses = 0
    for line_pairs in by_line.values():
        if len(line_pairs) < 2:
            continue
        (i_a, j_a), (i_b, j_b) = line_pairs[0], line_pairs[-1]
        words += i_b - i_a + 1
        span += hyp[j_b]["end"] - hyp[j_a]["start"]
        pauses += sum(hyp[j + 1]["start"] - hyp[j]["end"] >= PAUSE_SEC for j in range(j_a, j_b))
    wpm = words / span * 60 if span > 0 else 0
    fluency = 0.5 * scale(wpm, *WPM_RANGE) + 0.5 * max(0.0, 1 - PAUSE_COST * pauses)

    probs = [hyp[j]["p"] for _, j in pairs]
    confidence = sum(probs) / len(probs) if probs else 0
    clarity = scale(confidence, *CLARITY_RANGE)

    parts = {"accuracy": accuracy, "fluency": fluency, "clarity": clarity}
    points = {k: parts[k] * WEIGHTS[k] for k in WEIGHTS}
    total = sum(points.values())
    grade = next(g for cut, g in GRADES if total >= cut)
    return {"total": total, "grade": grade, "points": points, "accuracy": accuracy,
            "n_content": len(content), "n_names": len(ref) - len(content),
            "wpm": wpm, "pauses": pauses, "confidence": confidence}


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
    p.add_argument("--max-pass-words", type=int, default=3,
                   help="Only differences up to this many words can be passed as mis-hearing")
    p.add_argument("--no-boost", action="store_true", help="Don't raise quiet stretches")
    p.add_argument("--names", default="",
                   help="Extra names to leave out of accuracy, comma separated (e.g. Altman,Dario)")
    args = p.parse_args()

    ref = script_words(Path(args.script).read_text(encoding="utf-8"),
                       [n.strip() for n in args.names.split(",") if n.strip()])
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
        audio = load_audio(f)
        hyp = transcribe_words(model, audio, args)

        matched = 0
        rows = []
        correct = [False] * len(ref)
        pairs = []
        for op, rs, hs in compare(ref, hyp):
            n_ref = rs[1] - rs[0]
            if op == "equal":
                matched += n_ref
                correct[rs[0]:rs[1]] = [True] * n_ref
                if n_ref == hs[1] - hs[0]:
                    pairs += zip(range(*rs), range(*hs))
                else:  # "cyberattacks" vs "cyber attacks": only the ends line up
                    pairs += [(rs[0], hs[0]), (rs[1] - 1, hs[1] - 1)]
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
            elif (check and check["gap"] >= args.pass_gap
                  # A long stretch that differs is not a slip of the model's ear.
                  and max(n_ref, hs[1] - hs[0]) <= args.max_pass_words):
                verdict = "통과 (모델이 잘못 들음)"
                if op != "insert":
                    matched += n_ref
                    correct[rs[0]:rs[1]] = [True] * n_ref
            else:
                verdict = "확인 필요"
            rows.append({"t": t, "kind": kind, "script": script, "said": said,
                         "verdict": verdict, "check": check})

        rate = matched / len(ref) * 100 if ref else 0
        pairs.sort()
        s = score_reading(ref, hyp, correct, pairs)
        todo = sum(r["verdict"] == "확인 필요" for r in rows)
        summary.append((f.stem, rate, todo, s))

        lines = [f"# {f.stem}", ""]
        if s is None:
            lines += ["## 점수: **채점 불가** (목소리가 거의 들리지 않음 → 녹음 확인 필요)", ""]
        else:
            pt = s["points"]
            lines += [
                f"## 점수: **{s['total']:.0f}점 ({s['grade']})**", "",
                "| 항목 | 점수 | 근거 |", "|---|---|---|",
                f"| 정확도 | {pt['accuracy']:.0f}/{WEIGHTS['accuracy']} | 대본 단어 {s['accuracy'] * 100:.0f}% 맞게 읽음"
                f" (이름 {s['n_names']}개는 제외) |",
                f"| 유창성 | {pt['fluency']:.0f}/{WEIGHTS['fluency']} | 문장 안 속도 분당 {s['wpm']:.0f}단어,"
                f" {PAUSE_SEC}초 이상 멈춤 {s['pauses']}번 |",
                f"| 명확도 | {pt['clarity']:.0f}/{WEIGHTS['clarity']} | 맞게 읽은 단어의 평균 인식 확신도"
                f" {s['confidence']:.2f} |", ""]
        lines += [f"대본 일치율: **{rate:.0f}%** ({matched}/{len(ref)} 단어)", "",
                  "| 시간 | 유형 | 대본 | 아이가 읽은 것(받아쓰기) | 판정 | 점수 차 |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            gap = f"{r['check']['gap']:+.2f}" if r["check"] else "-"
            lines.append(f"| {r['t']:.1f}s | {r['kind']} | {r['script'] or '-'} | {r['said'] or '-'} "
                         f"| {r['verdict']} | {gap} |")
        lines += ["", "## 받아쓰기 전문", "", " ".join(w["text"] for w in hyp), ""]
        report = out_dir / f"{f.stem}.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"-> {report}")

    # One table for the whole class; utf-8-sig so Excel shows Korean correctly.
    csv_rows = ["이름,점수,등급,정확도,유창성,명확도,일치율(%),분당 단어,멈춤,확인 필요"]
    print("\n=== 요약 ===")
    for name, rate, todo, s in sorted(summary, key=lambda x: -(x[3]["total"] if x[3] else -1)):
        if s is None:
            print(f"{name}: 채점 불가 (녹음 확인)")
            csv_rows.append(f'"{name}",,채점 불가,,,,{rate:.0f},,,{todo}')
            continue
        pt = s["points"]
        print(f"{name}: {s['total']:.0f}점 {s['grade']}  (정확도 {pt['accuracy']:.0f} / 유창성 "
              f"{pt['fluency']:.0f} / 명확도 {pt['clarity']:.0f})  분당 {s['wpm']:.0f}단어, "
              f"멈춤 {s['pauses']}, 확신도 {s['confidence']:.2f}")
        csv_rows.append(f'"{name}",{s["total"]:.0f},{s["grade"]},{pt["accuracy"]:.1f},{pt["fluency"]:.1f},'
                        f'{pt["clarity"]:.1f},{rate:.0f},{s["wpm"]:.0f},{s["pauses"]},{todo}')
    (out_dir / "scores.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8-sig")
    print(f"-> {out_dir / 'scores.csv'}")


if __name__ == "__main__":
    main()

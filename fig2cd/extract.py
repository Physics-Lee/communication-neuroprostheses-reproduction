"""Calculate Figure 2(c,d) daily metrics from the authors' public releases.

Notes
-----
Each extractor reads the fields needed for scoring from one recording at a
time. The output contains daily aggregate counts, durations, and rates.
"""

import csv
import io
import json
import pickle
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import editdistance
import numpy as np
import openpyxl
from scipy.io import loadmat

if __package__:
    from .recompute import (
        COUNT_FIELDS, FIELDS, daily_rates_from_counts, format_daily_rates, summarize_daily,
    )
else:
    from recompute import (
        COUNT_FIELDS, FIELDS, daily_rates_from_counts, format_daily_rates, summarize_daily,
    )


WILLETT21_DAYS = ("2019.12.11", "2019.12.18", "2019.12.20", "2020.01.06", "2020.01.08")
WILLETT23_DAYS = ("2022.07.21", "2022.07.27", "2022.08.02", "2022.08.11", "2022.08.13")
CARD_DAYS = (27, 34, 46, 48, 69, 74, 76, 81, 83, 88, 90, 95, 223, 244)
KUNZ_DAYS = {
    "t12.2023.11.28_imagined_50words_eval.mat": "kunz25_t12",
    "t15.2024.12.15_imagined_125kwords_eval.mat": "kunz25_t15",
    "t15.2025.01.05_imagined_125kwords_eval.mat": "kunz25_t15",
    "t16.2024.12.18_imagined_125kwords_eval.mat": "kunz25_t16",
}
JUDE26B_BLOCKS = {
    "t17.2024.04.02": (13,),
    "t17.2024.05.07": (7, 8, 9, 10, 11),
    "t17.2024.05.14": (3, 6, 7, 8, 9),
    "t17.2024.05.28": (8, 9),
}
# Published vs. released trial counts for the three T17 free-typing blocks
# that differ. Use the available trials in each matched block.
JUDE26A_RELEASE_DIFFERENCES = {
    ("t17.2024.06.04", 9): (7, 5),
    ("t17.2024.07.09", 4): (8, 2),
    ("t17.2024.07.10", 3): (7, 8),
}
REDIS_FIELDS = (
    "cue", "go_cue_redis_time", "trial_end_redis_time", "block_description",
    "decoder_output_redis_clock", "ngram_decoder_final_output",
    "ngram_decoder_partial_output",
)
TOKEN_PUNCTUATION = re.compile(r"[^a-z0-9']+")


def text(value):
    """Read the first string from a MATLAB cell or array.

    Parameters
    ----------
    value : array_like
        MATLAB field value as loaded by SciPy, including scalar strings.

    Returns
    -------
    str
        The first flattened element stripped of surrounding whitespace if
        it is a string. Empty values and numeric placeholders yield an
        empty string.
    """
    flat = np.asarray(value).ravel()
    return str(flat[0]).strip() if flat.size and isinstance(flat[0], str) else ""


def display_text(value):
    """Convert release-specific handwriting symbols to display text.

    Parameters
    ----------
    value : array_like
        Raw MATLAB reference or decoder output.

    Returns
    -------
    str
        Stripped text with greater-than signs converted to spaces, tildes
        converted to periods, and hash signs removed.
    """
    return text(value).replace(">", " ").replace("~", ".").replace("#", "").strip()


def words(value):
    """Normalize a sentence for word edit-distance scoring.

    Parameters
    ----------
    value : str
        Reference or decoded sentence with release-specific symbols
        translated, if applicable.

    Returns
    -------
    list of str
        Case-folded, whitespace-delimited tokens. Each token retains only
        ASCII letters, digits, and apostrophes. Empty tokens are dropped.
    """
    stripped = (TOKEN_PUNCTUATION.sub("", token) for token in value.casefold().split())
    return [token for token in stripped if token]


class Day:
    """Accumulate trial counts, word edits, and elapsed time for one day.

    Parameters
    ----------
    unit : {"words", "characters"} or None
        Reference unit counted for speed. Use None when no speed is computed.

    Attributes
    ----------
    unit : str or None
        Reference unit for the speed count.
    released_trials : int
        Number of selected trials supplied to add.
    wer_trials, word_errors, reference_words : int
        Scored trial count, total word edits, and total reference words.
    speed_trials, speed_count : int
        Number of timed scored trials and total reference units in them.
    seconds : float
        Total duration of the timed scored trials, in seconds.
    """

    def __init__(self, unit):
        """Initialize the daily counters to zero."""
        self.unit = unit
        self.released_trials = 0
        self.wer_trials = self.word_errors = self.reference_words = 0
        self.speed_trials = self.speed_count = 0
        self.seconds = 0.0

    def add(self, reference, decoded, seconds=None):
        """Update the day's counts with one released trial.

        Parameters
        ----------
        reference : str
            Reference sentence used for the word and character counts.
        decoded : str or None
            Recorded decoder output. An empty string counts as a deletion
            of every reference word. None indicates an unavailable output.
        seconds : float or None, optional
            Trial duration in seconds. Only positive, finite durations from
            scored trials contribute to speed.

        Notes
        -----
        Every call increments the released-trial count. Scoring requires a
        nonempty normalized reference and an available decoder output.
        """
        self.released_trials += 1
        ref = words(reference)
        if not ref or decoded is None:
            return
        self.wer_trials += 1
        self.word_errors += editdistance.eval(ref, words(decoded))
        self.reference_words += len(ref)
        if self.unit and seconds is not None and np.isfinite(seconds) and seconds > 0:
            self.speed_trials += 1
            self.speed_count += len(reference) if self.unit == "characters" else len(ref)
            self.seconds += float(seconds)

    def record(self, series, day, source):
        """Serialize the counts and rates for one calendar day.

        Parameters
        ----------
        series : str
            Figure 2 series identifier, including the participant.
        day : str
            Calendar date in ISO YYYY-MM-DD format.
        source : str
            Description or path identifying the public inputs.

        Returns
        -------
        dict
            One row in the daily CSV schema. Speed fields are empty when
            the accumulator has no speed unit.

        Raises
        ------
        ValueError
            If the day has no scored trial, or speed was requested but no
            scored trial has a usable duration.
        """
        if not self.wer_trials:
            raise ValueError(f"No scored reference trials for {series} on {day}")
        if self.unit and not self.speed_trials:
            raise ValueError(f"No timed trials for {series} on {day}")
        speed = (
            self.speed_trials, self.speed_count, self.unit, f"{self.seconds:.9f}",
        ) if self.unit else ("", "", "", "")
        row = dict(zip(COUNT_FIELDS, (
            series, day, self.released_trials, self.wer_trials, self.word_errors, self.reference_words,
            *speed, source,
        ), strict=True))
        row.update(format_daily_rates(*daily_rates_from_counts(row, f"{series}, {day}")))
        return row


def willett21(root):
    """Extract the five published Willett 2021 copy-typing days.

    Parameters
    ----------
    root : pathlib.Path
        Released handwritingBCIData/Datasets directory containing the
        per-session sentences.mat files.

    Yields
    ------
    dict
        Daily counts for T5 after applying the released exclusion flags to
        the two closed-loop copy-typing conditions. Speed counts reference
        characters over the recorded 10 ms trial bins.
    """
    fields = (
        "sentenceCondition", "excludedSentences", "intendedText", "rnn_decodedText",
        "goCueOnsetTimeBin", "sentenceEndTimeBin",
    )
    for day in WILLETT21_DAYS:
        relative = f"t5.{day}/sentences.mat"
        data = loadmat(root / relative, variable_names=fields, squeeze_me=True)
        result = Day("characters")
        for condition, exclude, ref, hyp, go, end in zip(
            *(data[field] for field in fields), strict=True,
        ):
            if text(condition) in {"CL Corpus", "CL Pandarinath"} and not exclude:
                result.add(display_text(text(ref).replace(" ", "")),
                           display_text(hyp), (end - go) * 0.01)
        yield result.record("willett21_t5", day.replace(".", "-"), f"Datasets/{relative}")


def willett23(archive):
    """Extract the five vocal Willett 2023 large-vocabulary evaluation days.

    Parameters
    ----------
    archive : path-like
        Released sentences.tar.gz archive containing the session MAT files.

    Yields
    ------
    dict
        Daily counts for T12 from the closed-loop Switchboard Stage 2
        blocks. Speed counts reference words over the recorded 20 ms bins.

    Raises
    ------
    ValueError
        If a selected session is missing or has an unexpected speaking mode.

    Notes
    -----
    The archive is scanned sequentially and each selected MAT member is
    staged separately for SciPy's seekable file reader.
    """
    fields = (
        "sentences", "ngramFinalOutput", "sentenceDurations", "blockList",
        "blockTypes", "blockNum", "goTrialEpochs", "speakingMode",
    )
    wanted = {f"t12.{day}_sentences.mat": day for day in WILLETT23_DAYS}
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            filename = Path(member.name).name
            if not member.isfile() or filename not in wanted:
                continue
            # Stage one seekable MAT member on disk. The archive and its neural
            # arrays do not have to fit in memory.
            print(f"Reading {filename}", flush=True)
            with tempfile.TemporaryFile() as staged, source.extractfile(member) as entry:
                shutil.copyfileobj(entry, staged, length=8 * 1024 * 1024)
                staged.seek(0)
                data = loadmat(getattr(staged, "file", staged), variable_names=fields,
                               squeeze_me=True)
            if text(data["speakingMode"]) != "attempted speaking":
                raise ValueError(f"Expected vocal evaluation in {filename}")
            blocks = {int(n): text(t) for n, t in
                      zip(data["blockList"], data["blockTypes"], strict=True)}
            result = Day("words")
            for ref, hyp, duration, epoch in zip(
                data["sentences"], data["ngramFinalOutput"], data["sentenceDurations"],
                data["goTrialEpochs"], strict=True,
            ):
                block = int(data["blockNum"][int(epoch[0]) - 1])
                if blocks[block] == "CL Switchboard Stage 2":
                    result.add(text(ref), text(hyp), duration * 0.02)
            day = wanted.pop(filename).replace(".", "-")
            yield result.record("willett23_t12", day, f"sentences.tar.gz/{member.name}")
            if not wanted:
                break
    if wanted:
        raise ValueError(f"Missing Willett 2023 sessions: {', '.join(wanted)}")


def fan(archive):
    """Extract daily metrics from the released CORP recalibration arm.

    Parameters
    ----------
    archive : path-like
        Released CORP_data_release.zip archive.

    Yields
    ------
    dict
        Daily counts for T5 based on the recorded language model output.
        Speed uses reference characters and the 20 ms bins in tx_feats.

    Raises
    ------
    ValueError
        If the number of released online recalibration sessions differs
        from ten.
    """
    fields = ("sentences", "lm_decoded", "tx_feats")
    with zipfile.ZipFile(archive) as source:
        members = sorted(n for n in source.namelist()
                         if "/online_evaluation_data/recalibration/mat/" in n
                         and n.endswith(".mat"))
        for member in members:
            data = loadmat(io.BytesIO(source.read(member)), variable_names=fields,
                           squeeze_me=True)
            result = Day("characters")
            for ref, hyp, features in zip(*(data[k] for k in fields), strict=True):
                result.add(display_text(ref), display_text(hyp), len(features) * 0.02)
            day = Path(member).stem.removeprefix("t5.").replace(".", "-")
            yield result.record("fan23_t5", day, f"CORP_data_release.zip/{member}")
        if len(members) != 10:
            raise ValueError(f"Expected 10 released CORP sessions, found {len(members)}")


def card(path):
    """Extract the Card 2024 evaluation days listed in Table S3.

    Parameters
    ----------
    path : pathlib.Path
        Path to the authors' released t15_copyTask.pkl file.

    Yields
    ------
    dict
        Daily counts for T15 on the 14 large-vocabulary evaluation days,
        using the recorded speech durations for speed.
    """
    with path.open("rb") as stream:
        data = pickle.load(stream)
    days = {day: Day("words") for day in CARD_DAYS}
    for day, vocab, ref, hyp, seconds in zip(
        data["post_implant_day"], data["vocab_size"], data["cue_sentence"],
        data["decoded_sentence"], data["speech_duration_s"], strict=True,
    ):
        if vocab == 125000 and day in days:
            days[day].add(ref, hyp, seconds)
    for day, result in days.items():
        calendar_day = date(2023, 7, 17) + timedelta(days=day)
        yield result.record("card24_t15", calendar_day.isoformat(),
                            f"t15_copyTask.pkl, post-implant day {day}")


def kunz(root):
    """Extract the selected Kunz 2025 closed-loop inner-speech blocks.

    Parameters
    ----------
    root : pathlib.Path
        Extracted sentenceData directory containing the selected evaluation
        MAT files.

    Yields
    ------
    dict
        Daily WER counts for T12, T15, and T16 in the selected vocabulary
        conditions. Speed fields are empty.
    """
    fields = ("sentenceText", "decodedSentenceText")
    for filename, series in KUNZ_DAYS.items():
        data = loadmat(root / filename, variable_names=fields, squeeze_me=True)
        result = Day(None)
        # Each selected *_eval.mat contains only online evaluation trials.
        for ref, hyp in zip(*(data[k] for k in fields), strict=True):
            result.add(display_text(ref), display_text(hyp))
        day = filename[4:14].replace(".", "-")
        yield result.record(series, day, f"sentenceData.zip/sentenceData/{filename}")


def redis_blocks(root, participant, task, wanted):
    """Resolve selected RedisMat blocks to their released files.

    Parameters
    ----------
    root : pathlib.Path
        Extracted Data directory of a Jude dataset.
    participant : str
        Lowercase participant directory name, such as t17.
    task : str
        Task directory name, such as Typing or brainToText.
    wanted : set-like of tuple
        Requested (session, block_number) keys. Session names omit any
        trailing PU suffix.

    Returns
    -------
    dict
        Mapping from requested keys to their matching MAT file paths.

    Raises
    ------
    ValueError
        If any requested block has no file or resolves to multiple files.
    """
    paths = {}
    for path in sorted((root / participant).glob(f"*/{task}/RedisMat/*.mat")):
        match = re.search(r"\((\d+)\)\.mat$", path.name)
        if not match:
            continue
        key = path.parents[2].name.removesuffix("PU"), int(match[1])
        if key in wanted:
            if key in paths:
                raise ValueError(f"Multiple files for {key}: {paths[key]} and {path}")
            paths[key] = path
    if missing := wanted - paths.keys():
        raise ValueError(f"Missing RedisMat blocks for {participant}: {sorted(missing)}")
    return paths


def redis_trials(path):
    """Assign final decoder messages using the recorded trial and Redis clocks.

    Parameters
    ----------
    path : path-like
        Released RedisMat file for one evaluation block.

    Yields
    ------
    cue : str
        Reference sentence in display form.
    decoded : str or None
        Recovered decoder text. None means the tick-level record has no
        recoverable output.
    seconds : float
        Recorded duration from the go cue to trial end, in seconds.

    Raises
    ------
    ValueError
        If the decoder arrays do not fit a supported alignment or trial
        timestamp arrays and cues have different lengths.

    Notes
    -----
    Some blocks supply one final string per trial. In the tick-level format,
    choose the final closest to the trial end within the go-to-next-go
    interval. Fall back to the last visible partial inside the trial. A
    trial with no recoverable output record remains unscored.
    """
    m = loadmat(path, variable_names=REDIS_FIELDS)
    cues = [display_text(value) for value in m["cue"]]
    go = m["go_cue_redis_time"].ravel()
    end = m["trial_end_redis_time"].ravel()
    clock = m["decoder_output_redis_clock"].ravel()
    final = m["ngram_decoder_final_output"].ravel()
    partial = m["ngram_decoder_partial_output"].ravel()
    if len(final) not in {len(clock), len(cues)} or len(partial) != len(clock):
        raise ValueError(f"Unrecognized decoder output alignment in {path}")
    one_per_trial = len(final) == len(cues)
    final_ticks = [(t, s) for t, value in zip(clock, final)
                   if (s := display_text(value))] if not one_per_trial else []
    partial_ticks = [(t, s) for t, value in zip(clock, partial, strict=True)
                     if (s := display_text(value))]
    for i, (cue, start, stop) in enumerate(zip(cues, go, end, strict=True)):
        if one_per_trial:
            decoded = display_text(final[i])
        else:
            following = go[i + 1] if i + 1 < len(go) else np.inf
            candidates = ((t, s) for t, s in final_ticks if start <= t < following)
            _, decoded = min(candidates, key=lambda pair: abs(pair[0] - stop),
                             default=(None, None))
        if not decoded:
            decoded = next((s for t, s in reversed(partial_ticks) if start <= t <= stop), decoded)
        yield cue, decoded, (stop - start) / 1000


def jude26a(root, workbook):
    """Extract the released typing trials for the Jude 2026a Fig. 3b,c blocks.

    Parameters
    ----------
    root : pathlib.Path
        Extracted Data directory containing T17 and T18 Typing RedisMat files.
    workbook : pathlib.Path
        Published SourceData_Figure3.xlsx workbook used to select blocks.

    Yields
    ------
    dict
        Daily counts for T17 and T18, combining the selected blocks on the
        same calendar day. Speed uses reference characters.

    Raises
    ------
    ValueError
        If the workbook block totals are unexpected or a released trial count
        differs from the workbook outside the documented exceptions.
    """
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    try:
        for participant in ("T17", "T18"):
            blocks = Counter()
            for row in book[f"{participant}_Online_Decoding_Speeds"].values:
                match = re.fullmatch(r"(t\d+\.\d{4}\.\d{2}\.\d{2})(?:PU)?-Block-(\d+)",
                                     str(row[1]))
                if match:
                    blocks[match[1], int(match[2])] += 1
            if len(blocks) != (18 if participant == "T17" else 56):
                raise ValueError(f"Unexpected number of Figure 3 blocks for {participant}")
            paths = redis_blocks(root, participant.lower(), "Typing", blocks.keys())
            days = {}
            for (session, block), path in paths.items():
                trials = list(redis_trials(path))
                counts = blocks[session, block], len(trials)
                if counts[0] != counts[1] and counts != JUDE26A_RELEASE_DIFFERENCES.get(
                    (session, block)
                ):
                    raise ValueError(f"Unexpected published/released trial counts for {path}: {counts}")
                result = days.setdefault(session, Day("characters"))
                for trial in trials:
                    result.add(*trial)
            for session, result in days.items():
                day = session[4:].replace(".", "-")
                yield result.record(f"jude26a_{participant.lower()}", day,
                                    f"Data.zip/{participant.lower()}/{session}, Fig. 3 blocks")
    finally:
        book.close()


def jude26b(root):
    """Extract the Jude 2026b unconstrained-vocabulary evaluation blocks.

    Parameters
    ----------
    root : pathlib.Path
        Extracted Data directory containing the T17 brainToText RedisMat files.

    Yields
    ------
    dict
        Daily counts for T17. WER and speed use the same selected blocks on
        the four evaluation days. Speed uses reference words.
    """
    wanted = {(session, number) for session, blocks in JUDE26B_BLOCKS.items()
              for number in blocks}
    paths = redis_blocks(root, "t17", "brainToText", wanted)
    days = {}
    for (session, _), path in paths.items():
        result = days.setdefault(session, Day("words"))
        for trial in redis_trials(path):
            result.add(*trial)
    for session, result in days.items():
        yield result.record("jude26b_t17", session[4:].replace(".", "-"),
                            f"Data.zip/t17/{session}, Fig. 3B unconstrained vocabulary")


def rebuild(manifest, destination):
    """Rebuild and validate the daily CSV from the public releases.

    Parameters
    ----------
    manifest : path-like
        JSON mapping of the source keys listed in README.md to extracted
        directories or released files. Relative paths are resolved against
        the manifest's parent directory.
    destination : path-like
        Daily CSV path. Its parent directory must already exist.

    Notes
    -----
    Study groups follow paper publication dates. Within each study, rows
    are ordered by participant and calendar recording date. The destination
    is replaced after the staged CSV passes validation.
    """
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8") as stream:
        sources = {key: manifest.parent / value for key, value in json.load(stream).items()}
    # Keep procedures in the papers' publication order.
    procedures = (
        (willett21, ("willett21_datasets",)),
        (willett23, ("willett23_sentences_archive",)),
        (fan, ("fan23_archive",)),
        (card, ("card24_copy_task",)),
        (kunz, ("kunz25_sentence_data",)),
        (jude26a, ("jude26a_data", "jude26a_figure3_source_data")),
        (jude26b, ("jude26b_data",)),
    )
    rows = []
    for extract, keys in procedures:
        records = extract(*(sources[key] for key in keys))
        rows.extend(sorted(records, key=lambda row: (row["series"], row["day"])))
    # A failed extraction or invalid CSV leaves the last verified CSV in place.
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent, delete=False,
    ) as stream:
        staged = Path(stream.name)
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        summarize_daily(staged)
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)

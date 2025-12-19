########################
# Eval script for turn-taking (TT), user back-channeling (BC), user barge-in (BI)
# V2: Reads from output.jsonl format with inline audio paths and timestamps

import argparse
import json
import os
import re

import torch
import torchaudio
from nemo.collections import asr as nemo_asr
from tqdm import tqdm

INF_LATENCY = 9999.0


def parse_float_list(arg):
    """Parse a string representation of a list of floats."""
    if arg.startswith("[") and arg.endswith("]"):
        arg = arg[1:-1]
    return [float(x.strip()) for x in arg.split(",")]


def remove_special_symbols(text):
    """Remove special symbols like <SPECIAL_12> from text."""
    text = re.sub(r"<SPECIAL_\d+>", "", text)
    return text.strip()


def parse_timestamped_text(text_with_timestamps):
    """Parse BOS <|t|> and EOS <$t$> timestamps from text."""
    bos_pattern = r"<\|([\d\.]+)\|>"
    eos_pattern = r"<\$([\d\.]+)\$>"

    bos_matches = list(re.finditer(bos_pattern, text_with_timestamps))
    eos_matches = list(re.finditer(eos_pattern, text_with_timestamps))

    bos_timestamps = [float(match.group(1)) for match in bos_matches]
    eos_timestamps = [float(match.group(1)) for match in eos_matches]

    agent_segments = []

    if bos_timestamps and eos_timestamps:
        for i, start_time in enumerate(bos_timestamps):
            end_time = None
            eos_idx = None
            for j, eos_time in enumerate(eos_timestamps):
                if eos_time > start_time:
                    end_time = eos_time
                    eos_idx = j
                    break

            text = ""
            if end_time is not None and i < len(bos_matches) and eos_idx < len(eos_matches):
                bos_end_pos = bos_matches[i].end()
                eos_start_pos = eos_matches[eos_idx].start()
                text = text_with_timestamps[bos_end_pos:eos_start_pos].strip()

            if end_time is not None:
                agent_segments.append({"start": start_time, "end": end_time, "text": text})
            else:
                agent_segments.append({"start": start_time, "end": start_time + 5.0, "text": text})
    elif bos_timestamps:
        for i, timestamp in enumerate(bos_timestamps):
            text = ""
            if i < len(bos_matches):
                bos_end_pos = bos_matches[i].end()
                if i < len(bos_matches) - 1:
                    next_bos_start_pos = bos_matches[i + 1].start()
                    text = text_with_timestamps[bos_end_pos:next_bos_start_pos].strip()
                else:
                    text = text_with_timestamps[bos_end_pos:].strip()

            if i < len(bos_timestamps) - 1:
                agent_segments.append({"start": timestamp, "end": bos_timestamps[i + 1], "text": text})
            else:
                agent_segments.append({"start": timestamp, "end": timestamp + 5.0, "text": text})

    return agent_segments


def load_results_jsonl(results_dir, prefer_cached=True):
    """Load entries from output.jsonl or output_with_eval.jsonl in results directory.

    Args:
        results_dir: Directory containing the JSONL files
        prefer_cached: If True, prefer output_with_eval.jsonl if it exists (has cached segmentation/transcription)
    """
    cached_path = os.path.join(results_dir, "output_with_eval.jsonl")
    original_path = os.path.join(results_dir, "output.jsonl")

    # Prefer cached file if it exists and prefer_cached is True
    if prefer_cached and os.path.exists(cached_path):
        jsonl_path = cached_path
        print(f"Using cached results file: {jsonl_path}")
    else:
        jsonl_path = original_path

    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(data)

    return entries


def get_audio_dir(results_dir):
    """Get audio directory path from results directory."""
    return os.path.join(results_dir, "audio")


def _get_original_entry(entry):
    """Get the original entry, unwrapping if this is from output_with_eval.jsonl."""
    return entry.get("original_entry", entry)


def get_item_id(entry):
    """Extract item ID from entry."""
    entry = _get_original_entry(entry)
    audio_path = entry.get("audio_path", "")
    if audio_path:
        filename = os.path.basename(audio_path)
        if filename.endswith(".flac"):
            filename = filename[:-5]
        elif filename.endswith(".wav"):
            filename = filename[:-4]
        return filename
    return str(hash(json.dumps(entry, sort_keys=True)))[:8]


def get_audio_path(entry, audio_dir=None):
    """Get audio file path from entry, optionally remapping to audio_dir."""
    original = _get_original_entry(entry)
    audio = original.get("audio", {})
    if isinstance(audio, dict):
        path = audio.get("path", None)
        if path and audio_dir:
            # Remap to local audio directory
            filename = os.path.basename(path)
            return os.path.join(audio_dir, filename)
        return path
    return None


def get_generation_text(entry):
    """Get timestamped generation text from entry."""
    original = _get_original_entry(entry)
    generation = original.get("generation", "")
    if generation:
        return generation
    audio = original.get("audio", {})
    if isinstance(audio, dict):
        return audio.get("transcript", "")
    return ""


def get_cached_eval_data(entry):
    """Check if entry has cached segmentation and transcription from previous eval run.

    Returns tuple: (segmentation_data, transcription_data) or (None, None) if not cached.
    """
    # Check for eval data in the entry (from output_with_eval.jsonl format)
    segmentation = entry.get("segmentation", None)
    transcription = entry.get("transcription", None)

    if segmentation and transcription:
        return segmentation, transcription

    # Also check if this is an original_entry wrapper format
    original = entry.get("original_entry", None)
    if original:
        # This entry was already processed - use the eval data from parent
        segmentation = entry.get("segmentation", None)
        transcription = entry.get("transcription", None)
        if segmentation and transcription:
            return segmentation, transcription

    return None, None


def parse_cached_transcripts(transcription_dict):
    """Convert cached transcription format back to tuple-keyed dict.

    Cached format: {"1.234-5.678": "text"}
    Returns: {(1.234, 5.678): "text"}
    """
    result = {}
    for key, text in transcription_dict.items():
        try:
            start_str, end_str = key.split("-")
            result[(float(start_str), float(end_str))] = text
        except (ValueError, AttributeError):
            continue
    return result


def is_stopped_by_backchannel(agent_speech_segments, end_times, delay=0.99):
    """Check if agent's speech was interrupted by user backchanneling."""
    if not end_times or not agent_speech_segments:
        return [False] * len(end_times)

    agent_speech_segments = sorted(agent_speech_segments, key=lambda x: x["start"])

    bc_failure = []
    for t in end_times:
        if t == 0:
            bc_failure.append(False)
            continue

        overlapping = False
        for segment in agent_speech_segments:
            if t >= segment["start"] and t <= segment["end"]:
                overlapping = True
                agent_delayed_stop_time = t + delay
                agent_still_speaking = any(
                    s["start"] <= agent_delayed_stop_time <= s["end"] for s in agent_speech_segments
                )
                is_interrupted = not agent_still_speaking
                bc_failure.append(is_interrupted)
                break

        if not overlapping:
            bc_failure.append(False)

    return bc_failure


def find_user_barge_ins(user_turns, agent_turns, threshold_seconds=0.5):
    """Find user barge-in events during agent speech."""
    i, j = 0, 0
    success_barge_ins = []
    failed_barge_ins = []

    while i < len(user_turns) and j < len(agent_turns):
        u_start, u_end = user_turns[i]["start"], user_turns[i]["end"]
        a_start, a_end = agent_turns[j]["start"], agent_turns[j]["end"]

        if u_start > a_start and u_start < a_end:
            stop_duration_ms = round((a_end - u_start) * 1000)

            barge_in_info = {"stop_duration_ms": stop_duration_ms, "user": user_turns[i], "agent": agent_turns[j]}

            if stop_duration_ms < threshold_seconds * 1000:
                success_barge_ins.append(barge_in_info)
            else:
                failed_barge_ins.append(barge_in_info)

        if u_end < a_end:
            i += 1
        else:
            j += 1

    return success_barge_ins, failed_barge_ins


def init_vad_model():
    """Initialize Silero VAD model."""
    vad_model, utils = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False)
    vad_model = vad_model.to("cuda")
    get_speech_timestamps, _, _, _, _ = utils
    return vad_model, get_speech_timestamps


def init_asr_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    """Initialize NeMo ASR model."""
    print(f"Loading ASR model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name).cuda()
    print("ASR model loaded successfully.")
    return asr_model


def transcribe_segment(audio, start_time, end_time, sample_rate, asr_model, temp_dir="/tmp"):
    """Transcribe a specific segment of audio using NeMo ASR."""
    import tempfile

    try:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_audio = audio[:, start_sample:end_sample]

        if segment_audio.shape[1] < 160:
            return "", 0.0

        if sample_rate != 16000:
            segment_audio = torchaudio.functional.resample(segment_audio, sample_rate, 16000)
            sample_rate = 16000

        if segment_audio.shape[0] > 1:
            segment_audio = torch.mean(segment_audio, dim=0, keepdim=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmp_file:
            tmp_path = tmp_file.name
            torchaudio.save(tmp_path, segment_audio.cpu(), sample_rate)

        asr_outputs = asr_model.transcribe([tmp_path], timestamps=True)
        os.remove(tmp_path)

        if not asr_outputs:
            return "", 0.0

        result = asr_outputs[0]
        text = result.text if hasattr(result, "text") else ""

        end_timestamp = 0.0
        if hasattr(result, "timestamp") and result.timestamp and "word" in result.timestamp:
            word_timestamps = result.timestamp["word"]
            if word_timestamps and len(word_timestamps) > 0:
                last_word = word_timestamps[-1]
                if isinstance(last_word, dict) and "end" in last_word:
                    end_timestamp = last_word["end"]
        elif hasattr(result, "end_time"):
            end_timestamp = result.end_time

        return text.strip(), end_timestamp

    except Exception as e:
        print(f"Error transcribing segment [{start_time:.3f}s - {end_time:.3f}s]: {str(e)}")
        return "", 0.0


def compute_barge_in_metrics(success_barge_ins, failed_barge_ins):
    """Compute barge-in metrics including success rate and counts."""
    total_barge_ins = len(success_barge_ins) + len(failed_barge_ins)
    success_count = len(success_barge_ins)

    metrics = {"total_count": total_barge_ins, "success_count": success_count, "has_barge_ins": total_barge_ins > 0}

    if metrics["has_barge_ins"]:
        metrics["success_rate"] = (success_count / total_barge_ins) * 100

    if success_count > 0:
        metrics["avg_latency_ms"] = sum(bi["stop_duration_ms"] for bi in success_barge_ins) / success_count

    return metrics


def compute_turn_taking_metrics(
    agent_segments, user_segments, tt_latency_threshold_sec, tt_precision_buffer_sec, tt_recall_buffer_sec
):
    """Compute turn-taking metrics using precision and recall."""
    if not agent_segments or not user_segments:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_latency": INF_LATENCY,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    tp = 0
    fp = 0
    fn = 0
    tp_latencies = []

    for agent_seg in agent_segments:
        found_tp = False
        min_latency = INF_LATENCY

        for user_seg in user_segments:
            gap = agent_seg["start"] - user_seg["end"]
            if (
                gap >= -tt_precision_buffer_sec
                and gap <= tt_latency_threshold_sec
                and agent_seg["start"] >= user_seg["start"]
            ):
                found_tp = True
                min_latency = min(min_latency, max(gap, 0))

        if found_tp:
            tp += 1
            tp_latencies.append(min_latency)
        else:
            fp += 1

    for user_seg in user_segments:
        found_tp = False
        for agent_seg in agent_segments:
            gap = agent_seg["start"] - user_seg["end"]
            if gap >= -tt_recall_buffer_sec and gap <= tt_latency_threshold_sec:
                found_tp = True
                break
        if not found_tp:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = sum(tp_latencies) / len(tp_latencies) if tp_latencies else INF_LATENCY

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency": avg_latency,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def print_detailed_utterance(metrics_dict):
    """Print detailed information for a single utterance."""
    user_transcripts = metrics_dict.get("user_transcripts", {})
    agent_transcripts = metrics_dict.get("agent_transcripts", {})

    print(f"\n{'=' * 80}")
    print(f"Utterance: {metrics_dict['item_id']}")
    print(f"{'=' * 80}")

    print("\nMetrics:")
    print("  Turn-taking:")
    print(f"    - Precision: {metrics_dict['tt_precision']:.3f}")
    print(f"    - Recall: {metrics_dict['tt_recall']:.3f}")
    print(f"    - F1: {metrics_dict['tt_f1']:.3f}")
    print(f"    - Latency: {metrics_dict['tt_latency']:.3f}s ({metrics_dict['tt_latency'] * 1000:.1f}ms)")

    if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
        print("  Barge-in:")
        print(
            f"    - Success rate: {metrics_dict['barge_in_metrics']['success_rate']:.1f}% ({metrics_dict['barge_in_metrics']['success_count']}/{metrics_dict['barge_in_metrics']['total_count']})"
        )
        if "avg_latency_ms" in metrics_dict["barge_in_metrics"]:
            print(f"    - Average latency: {metrics_dict['barge_in_metrics']['avg_latency_ms']:.1f}ms")
    else:
        print("  Barge-in: No barge-ins detected")

    print("\nConversation flow:")
    all_segments = []
    for seg in metrics_dict["user_segments"]:
        all_segments.append({"type": "User", "start": seg["start"], "end": seg["end"]})
    for seg in metrics_dict["agent_segments"]:
        all_segments.append({"type": "Agent", "start": seg["start"], "end": seg["end"]})

    all_segments.sort(key=lambda x: x["start"])

    for seg in all_segments:
        seg_key = (seg["start"], seg["end"])
        duration = seg["end"] - seg["start"]

        if seg["type"] == "User":
            transcript = user_transcripts.get(seg_key, "")
            if transcript:
                print(
                    f"  \033[94mUser\033[0m  [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s): {transcript}"
                )
            else:
                print(f"  \033[94mUser\033[0m  [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s)")
        else:
            transcript = agent_transcripts.get(seg_key, "")
            if transcript:
                cleaned_text = remove_special_symbols(transcript)
                print(
                    f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s): {cleaned_text}"
                )
            else:
                print(f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s)")

    if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
        print("\nBarge-in events:")
        if metrics_dict["success_barge_ins"]:
            print(f"  Successful ({len(metrics_dict['success_barge_ins'])}):")
            for bi in metrics_dict["success_barge_ins"]:
                print(f"    User barged in at {bi['user']['start']:.3f}s during agent speech")
                print(f"    Agent stopped in {bi['stop_duration_ms']:.1f}ms")

        if metrics_dict["failed_barge_ins"]:
            print(f"  Failed ({len(metrics_dict['failed_barge_ins'])}):")
            for bi in metrics_dict["failed_barge_ins"]:
                print(f"    User barged in at {bi['user']['start']:.3f}s during agent speech")
                print(f"    Agent took {bi['stop_duration_ms']:.1f}ms to stop (too slow)")

    print(f"{'=' * 80}\n")


def print_bottom_percentile_utterances(all_metrics, percentile=5):
    """Print utterances in the bottom percentile for barge-in accuracy and turn-taking recall."""
    import numpy as np

    if not all_metrics:
        print("No metrics to analyze.")
        return

    print(f"\n{'#' * 80}")
    print(f"BOTTOM {percentile}% PERCENTILE ANALYSIS")
    print(f"{'#' * 80}\n")

    utterances_with_barge_ins = [m for m in all_metrics if m["barge_in_metrics"]["has_barge_ins"]]
    barge_in_rates = [m["barge_in_metrics"]["success_rate"] for m in utterances_with_barge_ins]
    tt_recalls = [m["tt_recall"] for m in all_metrics]

    if barge_in_rates:
        barge_in_threshold = np.percentile(barge_in_rates, percentile)
        print(f"Barge-in success rate {percentile}th percentile threshold: {barge_in_threshold:.1f}%")
        print(f"  (Based on {len(utterances_with_barge_ins)} utterances with barge-ins)\n")
    else:
        barge_in_threshold = None
        print("No utterances with barge-ins detected.\n")

    tt_recall_threshold = np.percentile(tt_recalls, percentile)
    print(f"Turn-taking recall {percentile}th percentile threshold: {tt_recall_threshold:.3f}")
    print(f"  (Based on {len(all_metrics)} utterances)\n")

    low_barge_in_utterances = []
    if barge_in_threshold is not None:
        low_barge_in_utterances = [
            m for m in utterances_with_barge_ins if m["barge_in_metrics"]["success_rate"] <= barge_in_threshold
        ]
        low_barge_in_utterances.sort(key=lambda m: m["barge_in_metrics"]["success_rate"])

    low_tt_recall_utterances = [m for m in all_metrics if m["tt_recall"] <= tt_recall_threshold]
    low_tt_recall_utterances.sort(key=lambda m: m["tt_recall"])

    if low_barge_in_utterances:
        print(f"\n{'-' * 80}")
        print(f"UTTERANCES WITH LOW BARGE-IN SUCCESS RATE (≤ {barge_in_threshold:.1f}%)")
        print(f"Found {len(low_barge_in_utterances)} utterance(s)")
        print(f"{'-' * 80}")

        for m in low_barge_in_utterances:
            print_detailed_utterance(m)

    if low_tt_recall_utterances:
        print(f"\n{'-' * 80}")
        print(f"UTTERANCES WITH LOW TURN-TAKING RECALL (≤ {tt_recall_threshold:.3f})")
        print(f"Found {len(low_tt_recall_utterances)} utterance(s)")
        print(f"{'-' * 80}")

        for m in low_tt_recall_utterances:
            print_detailed_utterance(m)

    print(f"\n{'#' * 80}")
    print(f"END OF BOTTOM {percentile}% PERCENTILE ANALYSIS")
    print(f"{'#' * 80}\n")


def print_metrics(metrics_dict, verbose=False):
    """Print all evaluation metrics for a conversation."""
    if not verbose:
        return

    barge_in_stats = []
    if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
        barge_in_stats.append(
            f"   - Barge-in success rate: {metrics_dict['barge_in_metrics']['success_rate']:.1f}% ({metrics_dict['barge_in_metrics']['success_count']}/{metrics_dict['barge_in_metrics']['total_count']})"
        )
        if "avg_latency_ms" in metrics_dict["barge_in_metrics"]:
            barge_in_stats.append(
                f"   - Average barge-in latency: {metrics_dict['barge_in_metrics']['avg_latency_ms']:.1f} ms"
            )
    else:
        barge_in_stats.append("   - No barge-ins detected")

    segment_info = ""
    if verbose:
        user_transcripts = metrics_dict.get("user_transcripts", {})
        agent_transcripts = metrics_dict.get("agent_transcripts", {})

        all_segments = []
        for seg in metrics_dict["user_segments"]:
            all_segments.append({"type": "User", "start": seg["start"], "end": seg["end"]})
        for seg in metrics_dict["agent_segments"]:
            all_segments.append({"type": "Agent", "start": seg["start"], "end": seg["end"]})

        all_segments.sort(key=lambda x: x["start"])

        user_to_agent_latencies = {}
        for user_seg in metrics_dict["user_segments"]:
            next_agent = None
            min_latency = float("inf")
            for agent_seg in metrics_dict["agent_segments"]:
                if agent_seg["start"] >= user_seg["end"]:
                    latency = agent_seg["start"] - user_seg["end"]
                    if latency < min_latency:
                        min_latency = latency
                        next_agent = agent_seg

            if next_agent:
                user_to_agent_latencies[(user_seg["start"], user_seg["end"])] = min_latency

        def format_segment(seg):
            seg_key = (seg["start"], seg["end"])
            transcript = ""

            if seg["type"] == "User":
                if seg_key in user_transcripts:
                    transcript = f" ({user_transcripts[seg_key]})"

                if seg_key in user_to_agent_latencies:
                    latency = user_to_agent_latencies[seg_key]
                    return f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s], \033[93m{latency:.3f}s\033[0m{transcript}"
                else:
                    return (
                        f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{transcript}"
                    )
            else:
                if seg_key in agent_transcripts:
                    cleaned_text = remove_special_symbols(agent_transcripts[seg_key])
                    words = cleaned_text.split()
                    estimate_sec_per_word = 0.3
                    estimated_duration = len(words) * estimate_sec_per_word
                    transcript = f" ({cleaned_text}) [\033[95mest. {estimated_duration:.2f}s\033[0m]"
                return f"   \033[92m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{transcript}"

        segments_str = "\n".join(format_segment(seg) for seg in all_segments)

        barge_in_segments = []
        if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
            if metrics_dict["success_barge_ins"]:
                barge_in_segments.append("   Successful barge-ins:")
                for bi in metrics_dict["success_barge_ins"]:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.3f}s - {bi['user']['end']:.3f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.3f}s - {bi['agent']['end']:.3f}s]")
                    barge_in_segments.append(f"     Stop duration: {bi['stop_duration_ms']:.3f} ms")

            if metrics_dict["failed_barge_ins"]:
                barge_in_segments.append("   Failed barge-ins:")
                for bi in metrics_dict["failed_barge_ins"]:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.3f}s - {bi['user']['end']:.3f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.3f}s - {bi['agent']['end']:.3f}s]")
                    barge_in_segments.append(f"     Stop duration: {bi['stop_duration_ms']:.3f} ms")

        segment_info = f"""
4. Speech segments (chronological order):
{segments_str}
5. Barge-in details:
{chr(10).join(barge_in_segments) if barge_in_segments else "   No barge-ins detected"}"""

    output = f"""
Evaluation metrics for conversation {metrics_dict["item_id"]}:
1. Turn-taking metrics:
   - Average latency: {metrics_dict["tt_latency"]:.3f} seconds
   - Precision: {metrics_dict["tt_precision"]:.3f}
   - Recall: {metrics_dict["tt_recall"]:.3f}
   - F1: {metrics_dict["tt_f1"]:.3f}
2. Barge-in statistics:
{chr(10).join(barge_in_stats)}
3. Backchanneling failures: {metrics_dict["bc_failure"]}{segment_info}
{"-" * 50}"""

    print(output)


def main(args):
    print(f"Loading results from: {args.results_dir}")
    # Load entries, preferring cached results unless force_recompute is set
    entries = load_results_jsonl(args.results_dir, prefer_cached=not args.force_recompute)
    audio_dir = get_audio_dir(args.results_dir)
    print(f"Loaded {len(entries)} entries")
    print(f"Audio directory: {audio_dir}")

    # Initialize models (may not be needed if all entries have cached data)
    vad_model, get_speech_timestamps = init_vad_model()

    asr_model = None
    if not args.disable_transcription:
        asr_model = init_asr_model(args.asr_model_name)

    # Metrics accumulators
    count = 0
    all_tt_latencies = []
    all_tt_precisions = []
    all_tt_recalls = []
    all_tt_f1s = []
    all_barge_in_success_rates = []
    all_barge_in_latencies = []
    all_bc_accuracies = []
    all_metrics_dicts = []

    # For per-sample results output
    per_sample_results = []

    for idx, entry in enumerate(tqdm(entries, desc="Processing entries")):
        item_id = get_item_id(entry)
        audio_path = get_audio_path(entry, audio_dir=audio_dir)
        generation_text = get_generation_text(entry)

        # Check for cached segmentation and transcription data (unless force_recompute)
        cached_segmentation, cached_transcription = get_cached_eval_data(entry)
        use_cached = not args.force_recompute and cached_segmentation is not None and cached_transcription is not None

        # Get audio filename for logging
        audio_filename = os.path.basename(audio_path) if audio_path else "unknown"

        if use_cached:
            print(f"\nProcessing: {item_id}")
            print(f"  Audio: {audio_filename}")
            print("  Using cached segmentation/transcription")
            # Extract segments from cached data
            user_segments = cached_segmentation.get("user_segments", [])
            agent_segments = cached_segmentation.get("agent_segments", [])
            use_vad_for_agent = cached_segmentation.get("used_audio_segmentation_for_agent", False)

            # Extract transcripts from cached data
            user_transcripts = parse_cached_transcripts(cached_transcription.get("user", {}))
            agent_transcripts = parse_cached_transcripts(cached_transcription.get("agent", {}))

            print(f"  Cached: {len(user_segments)} user segments, {len(agent_segments)} agent segments")

            # Get original entry if this is a wrapper format
            original_entry = entry.get("original_entry", entry)
        else:
            if not audio_path or not os.path.exists(audio_path):
                print(f"Skipping {item_id}: audio file not found at {audio_path}")
                continue

            print(f"\nProcessing: {item_id}")
            print(f"  Audio: {audio_filename} ({audio_path})")

            # Load stereo audio: channel 0 = user, channel 1 = agent
            audio, audio_sr = torchaudio.load(audio_path)
            if audio.shape[0] < 2:
                print(f"  Warning: Expected stereo audio, got {audio.shape[0]} channel(s). Skipping.")
                continue

            user_audio = audio[0:1, :]
            agent_audio = audio[1:2, :]

            # Resample to 16kHz for VAD
            user_audio_16k = torchaudio.functional.resample(user_audio, audio_sr, 16000)
            agent_audio_16k = torchaudio.functional.resample(agent_audio, audio_sr, 16000)

            # Get agent segments from timestamps or VAD fallback
            agent_segments = []
            use_vad_for_agent = args.use_audio_segmentation

            if not use_vad_for_agent and generation_text:
                agent_segments = parse_timestamped_text(generation_text)
                if agent_segments:
                    print(f"  Parsed {len(agent_segments)} agent segments from timestamps")

            if not agent_segments:
                if args.use_audio_segmentation:
                    print("  Using VAD for agent segmentation (--use_audio_segmentation)")
                else:
                    print("  No timestamps found, using VAD for agent segmentation")
                agent_vad_results = get_speech_timestamps(
                    agent_audio_16k.to("cuda"),
                    vad_model,
                    sampling_rate=16000,
                    min_silence_duration_ms=args.vad_min_silence_duration_ms,
                )
                agent_segments = [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in agent_vad_results]
                use_vad_for_agent = True
                print(f"  VAD detected {len(agent_segments)} agent segments")

            # Get user segments via VAD
            user_vad_results = get_speech_timestamps(
                user_audio_16k.to("cuda"),
                vad_model,
                sampling_rate=16000,
                min_silence_duration_ms=args.vad_min_silence_duration_ms,
            )
            user_segments = [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in user_vad_results]
            print(f"  VAD detected {len(user_segments)} user segments")

            # Initialize transcripts (will be filled below if not cached)
            user_transcripts = {}
            agent_transcripts = {}
            original_entry = entry

        # Compute metrics
        tt_metrics = compute_turn_taking_metrics(
            agent_segments,
            user_segments,
            args.tt_latency_threshold_sec,
            args.tt_precision_buffer_sec,
            args.tt_recall_buffer_sec,
        )
        tt_latency = tt_metrics["avg_latency"]

        success_barge_ins, failed_barge_ins = find_user_barge_ins(
            user_segments, agent_segments, args.barge_in_threshold_sec
        )
        barge_in_metrics = compute_barge_in_metrics(success_barge_ins, failed_barge_ins)

        end_time = args.end_time
        if end_time is not None:
            bc_failure = is_stopped_by_backchannel(agent_segments, end_time, args.barge_in_threshold_sec)
        else:
            bc_failure = []

        # Store metrics
        all_tt_latencies.append(tt_latency)
        all_tt_precisions.append(tt_metrics["precision"])
        all_tt_recalls.append(tt_metrics["recall"])
        all_tt_f1s.append(tt_metrics["f1"])

        if barge_in_metrics["has_barge_ins"]:
            all_barge_in_success_rates.append(barge_in_metrics["success_rate"])
            if "avg_latency_ms" in barge_in_metrics:
                all_barge_in_latencies.append(barge_in_metrics["avg_latency_ms"])

        bc_accuracy = sum(1 for x in bc_failure if not x) / len(bc_failure) if bc_failure else 0
        all_bc_accuracies.append(bc_accuracy)

        # Transcriptions - skip if using cached data
        if not use_cached:
            # Extract agent text from timestamps OR transcribe with ASR
            if use_vad_for_agent and asr_model is not None:
                # When using audio segmentation, transcribe agent segments with ASR
                print(f"  Transcribing {len(agent_segments)} agent segments with ASR...")
                for seg in agent_segments:
                    transcript, _ = transcribe_segment(agent_audio, seg["start"], seg["end"], audio_sr, asr_model)
                    agent_transcripts[(seg["start"], seg["end"])] = transcript
            elif generation_text and not use_vad_for_agent:
                # Extract text from timestamp markers
                agent_segments_with_text = parse_timestamped_text(generation_text)
                agent_transcripts = {
                    (seg["start"], seg["end"]): seg.get("text", "") for seg in agent_segments_with_text
                }

            # Transcribe user segments with ASR
            if asr_model is not None:
                print(f"  Transcribing {len(user_segments)} user segments...")
                for seg in user_segments:
                    transcript, _ = transcribe_segment(user_audio, seg["start"], seg["end"], audio_sr, asr_model)
                    user_transcripts[(seg["start"], seg["end"])] = transcript

        metrics_dict = {
            "item_id": item_id,
            "tt_latency": tt_latency,
            "tt_accuracy": tt_metrics["f1"],
            "tt_precision": tt_metrics["precision"],
            "tt_recall": tt_metrics["recall"],
            "tt_f1": tt_metrics["f1"],
            "barge_in_metrics": barge_in_metrics,
            "bc_failure": bc_failure,
            "user_segments": user_segments,
            "agent_segments": agent_segments,
            "success_barge_ins": success_barge_ins,
            "failed_barge_ins": failed_barge_ins,
            "user_transcripts": user_transcripts,
            "agent_transcripts": agent_transcripts,
        }

        all_metrics_dicts.append(metrics_dict)
        print_metrics(metrics_dict, verbose=args.verbose)
        count += 1

        # Build per-sample result for output
        if args.save_per_sample_results:
            # Convert tuple keys to string for JSON serialization
            user_transcripts_json = {f"{k[0]:.3f}-{k[1]:.3f}": v for k, v in user_transcripts.items()}
            agent_transcripts_json = {f"{k[0]:.3f}-{k[1]:.3f}": v for k, v in agent_transcripts.items()}

            sample_result = {
                "original_entry": original_entry,
                "eval_metrics": {
                    "turn_taking": {
                        "latency": tt_latency,
                        "precision": tt_metrics["precision"],
                        "recall": tt_metrics["recall"],
                        "f1": tt_metrics["f1"],
                    },
                    "barge_in": {
                        "has_barge_ins": barge_in_metrics["has_barge_ins"],
                        "success_rate": barge_in_metrics.get("success_rate", 0),
                        "success_count": barge_in_metrics.get("success_count", 0),
                        "total_count": barge_in_metrics.get("total_count", 0),
                        "avg_latency_ms": barge_in_metrics.get("avg_latency_ms", None),
                    },
                    "backchanneling": {"failures": bc_failure},
                },
                "segmentation": {
                    "user_segments": [{"start": s["start"], "end": s["end"]} for s in user_segments],
                    "agent_segments": [{"start": s["start"], "end": s["end"]} for s in agent_segments],
                    "used_audio_segmentation_for_agent": use_vad_for_agent,
                },
                "transcription": {"user": user_transcripts_json, "agent": agent_transcripts_json},
            }
            per_sample_results.append(sample_result)

    # Compute and print average metrics
    _valid_tt_latencies = [x for x in all_tt_latencies if x != INF_LATENCY]
    avg_metrics = {
        "avg_tt_latency": sum(_valid_tt_latencies) / len(_valid_tt_latencies) if _valid_tt_latencies else 0,
        "avg_tt_precision": sum(all_tt_precisions) / len(all_tt_precisions) * 100 if all_tt_precisions else 0,
        "avg_tt_recall": sum(all_tt_recalls) / len(all_tt_recalls) * 100 if all_tt_recalls else 0,
        "avg_tt_f1": sum(all_tt_f1s) / len(all_tt_f1s) * 100 if all_tt_f1s else 0,
        "avg_barge_in_success_rate": sum(all_barge_in_success_rates) / len(all_barge_in_success_rates)
        if all_barge_in_success_rates
        else 0,
        "avg_barge_in_latency": sum(all_barge_in_latencies) / len(all_barge_in_latencies)
        if all_barge_in_latencies
        else 0,
        "avg_bc_accuracy": sum(all_bc_accuracies) / len(all_bc_accuracies) * 100 if all_bc_accuracies else 0,
        "num_audios_evaluated": count,
    }

    avg_metrics_str = f"""
{"=" * 50}
Average Metrics:
1. Turn-taking:
   - Average latency: {avg_metrics["avg_tt_latency"] * 1000:.1f} ms
   - Precision: {avg_metrics["avg_tt_precision"]:.1f}%
   - Recall: {avg_metrics["avg_tt_recall"]:.1f}%
   - F1: {avg_metrics["avg_tt_f1"]:.1f}%
2. User barge-in:
   - Average success rate: {avg_metrics["avg_barge_in_success_rate"]:.1f}%
   - Average latency: {avg_metrics["avg_barge_in_latency"]:.1f} ms
3. Back-channeling:
   - Average accuracy: {avg_metrics["avg_bc_accuracy"]:.1f}%
4. Number of audios evaluated: {avg_metrics["num_audios_evaluated"]}
{"=" * 50}"""

    print(avg_metrics_str)

    if args.show_bottom_percentile:
        print_bottom_percentile_utterances(all_metrics_dicts, percentile=args.percentile_threshold)

    # Save dataset-level metrics to JSON file (in results_dir)
    if args.output_file:
        # If not an absolute path, save in results_dir
        if not os.path.isabs(args.output_file):
            output_file_path = os.path.join(args.results_dir, args.output_file)
        else:
            output_file_path = args.output_file

        dataset_metrics = {
            "dataset_metrics": {
                "turn_taking": {
                    "avg_latency_ms": avg_metrics["avg_tt_latency"] * 1000,
                    "avg_precision": avg_metrics["avg_tt_precision"],
                    "avg_recall": avg_metrics["avg_tt_recall"],
                    "avg_f1": avg_metrics["avg_tt_f1"],
                },
                "barge_in": {
                    "avg_success_rate": avg_metrics["avg_barge_in_success_rate"],
                    "avg_latency_ms": avg_metrics["avg_barge_in_latency"],
                },
                "backchanneling": {"avg_accuracy": avg_metrics["avg_bc_accuracy"]},
                "num_samples_evaluated": avg_metrics["num_audios_evaluated"],
            },
            "args": vars(args),
        }
        with open(output_file_path, "w") as f:
            json.dump(dataset_metrics, f, indent=2)
        print(f"\nDataset metrics saved to: {output_file_path}")

    # Save per-sample results to JSONL
    if args.save_per_sample_results and per_sample_results:
        output_jsonl_path = os.path.join(args.results_dir, "output_with_eval.jsonl")
        with open(output_jsonl_path, "w") as f:
            for result in per_sample_results:
                f.write(json.dumps(result) + "\n")
        print(f"Per-sample results saved to: {output_jsonl_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate conversation behavior from output.jsonl format")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory containing output.jsonl and audio/ subdirectory"
    )
    parser.add_argument(
        "--barge_in_threshold_sec",
        type=float,
        default=1.5,
        help="Buffering time for the agent to stop after user barges in",
    )
    parser.add_argument(
        "--tt_latency_threshold_sec",
        type=float,
        default=1.5,
        help="Threshold in seconds for considering turn-taking accurate",
    )
    parser.add_argument(
        "--tt_precision_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for precision calculation"
    )
    parser.add_argument(
        "--tt_recall_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for recall calculation"
    )
    parser.add_argument(
        "--end_time",
        type=lambda x: None if x is None or x.lower() == "none" else parse_float_list(x),
        default=None,
        help="End time of backchanneling. Format: '[1,10,15,20]' or '1,10,15,20' or None",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed segment information")
    parser.add_argument(
        "--vad_min_silence_duration_ms",
        type=int,
        default=1500,
        help="Minimum silence duration in milliseconds for VAD",
    )
    parser.add_argument(
        "--disable_transcription",
        action="store_true",
        default=False,
        help="Disable ASR transcription of user segments (enabled by default)",
    )
    parser.add_argument(
        "--asr_model_name",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Name of the ASR model for user transcription",
    )
    parser.add_argument(
        "--show_bottom_percentile",
        action="store_true",
        default=True,
        help="Show analysis of utterances in the bottom percentile",
    )
    parser.add_argument(
        "--percentile_threshold",
        type=float,
        default=5.0,
        help="Percentile threshold for identifying low-quality utterances",
    )
    parser.add_argument(
        "--use_audio_segmentation",
        action="store_true",
        default=False,
        help="Use VAD+ASR for both channels instead of text timestamps. "
        "This ignores <|t|> markers and derives all timing from audio analysis.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics.json",
        help="Filename for dataset-level metrics JSON (saved in results_dir)",
    )
    parser.add_argument(
        "--save_per_sample_results",
        action="store_true",
        default=False,
        help="Save per-sample segmentation, ASR, and metrics to output_with_eval.jsonl",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Force recompute segmentation and ASR even if cached results exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

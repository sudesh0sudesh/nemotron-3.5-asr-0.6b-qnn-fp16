#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Transcribe WAV audio using only the compiled QNN HTP neural graphs."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
import wave
import numpy as np
from runtime import NativeEncoder, NativeGraph, transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="16 kHz mono PCM16 WAV")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be nonnegative and repetitions must be positive")
    with wave.open(str(args.audio), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
            parser.error("Input must be 16 kHz, mono, PCM16 WAV")
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if not len(audio):
        parser.error("Audio is empty")
    root = args.model_dir
    vocab = json.loads((root / "vocab.json").read_text())
    languages = json.loads((root / "languages.json").read_text())["promptDictionary"]
    if args.language not in languages:
        parser.error("Unknown language prompt: " + args.language)
    with ExitStack() as stack:
        started = time.perf_counter()
        encoder = NativeEncoder(root / "models/encoder.bin")
        stack.callback(encoder.context.close)
        decoder = NativeGraph(root / "models/decoder.bin", ["token", "h", "c"], ["decoder_output", "h_out", "c_out"])
        stack.callback(decoder.context.close)
        joint = NativeGraph(root / "models/joint.bin", ["encoder_output", "decoder_output"], ["token_id"])
        stack.callback(joint.context.close)
        data = {"backend": "qnn-htp-only", "precision": "fp16", "language": args.language,
                "audio_seconds": len(audio) / 16000, "initialization_ms": (time.perf_counter() - started) * 1000,
                "warmups": args.warmups, "repetitions": args.repetitions, "runs": []}
        for repetition in range(args.warmups + args.repetitions):
            started = time.perf_counter()
            result = transcribe(encoder, decoder, joint, vocab, languages[args.language], audio)
            result["end_to_end_ms"] = (time.perf_counter() - started) * 1000
            result.update(repetition=repetition, warmup=repetition < args.warmups)
            if result["symbol_limit_hits"]:
                raise RuntimeError("Greedy decoding hit the symbol limit; inspect the input")
            data["runs"].append(result)
            print("TRANSCRIPTION:" + json.dumps({k: result[k] for k in ["text", "end_to_end_ms", "warmup"]}), flush=True)
            for graph in [encoder, decoder, joint]:
                graph.context.samples.clear()
        times = [r["end_to_end_ms"] for r in data["runs"] if not r["warmup"]]
        data["summary"] = {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
                           "times_real_time": data["audio_seconds"] * 1000 / statistics.mean(times)}
        data["libraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libQnn" in line})
        if any("libQnnCpu" in name or "libQnnGpu" in name for name in data["libraries"]):
            raise RuntimeError("Unexpected QNN backend loaded")
        if args.output:
            args.output.write_text(json.dumps(data, indent=2) + "\n")
        print("RESULT_JSON:" + json.dumps(data), flush=True)


if __name__ == "__main__":
    main()

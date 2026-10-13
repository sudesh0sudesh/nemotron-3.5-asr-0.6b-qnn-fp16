
# Nemotron 3.5 ASR · FP16 · Qualcomm NPU

**A complete neural-inference port of NVIDIA Nemotron 3.5 ASR 0.6B to the Arduino VENTUNO Q's Hexagon V75 NPU.**

The encoder, recurrent prediction decoder, joint network and ArgMax token selection execute on **QNN HTP**. This release contains three compiled NPU context binaries and an HTP-only runner. It does not require ONNX Runtime, PyTorch or LiteRT, and has no CPU/GPU neural fallback.

Audio preprocessing, cache-buffer management, the greedy control loop and token-to-text formatting execute on CPU. “All-NPU” describes neural inference, not a CPU-free application.

**Full-corpus validation performed on the Arduino:** **8.35% English FLEURS WER**, compared with the original model card's **8.27%** at the matching 320 ms setting, and **7.29× real-time processing** across 106.5 minutes of audio.

## Hardware and runtime

| Property | Target |
|---|---|
| Board | **Arduino VENTUNO Q** |
| SoC | Qualcomm IQ/QCS8275, SoC ID **82** |
| Accelerator | Hexagon **V75** / QNN HTP |
| OS / host architecture | Tested on Ubuntu 24.04, Linux ARM64 |
| Compiled precision | **FP16** neural computation; float32/int32 graph I/O |
| Compile / tested runtime | QAIRT **2.45** / **2.46.0.260424121129** |
| Streaming configuration | **320 ms**, attention context `[56,3]`, 240 ms lookahead |
| Input | 16 kHz mono audio |

The context binaries are target-specific. They are not portable ONNX files and do not run on arbitrary Snapdragon, desktop CPU/GPU or other Hexagon generations.

## Quick start on the Arduino

### GitHub checkout

```sh
git clone https://github.com/sudesh0sudesh/nemotron-3.5-asr-0.6b-qnn-fp16.git
cd nemotron-3.5-asr-0.6b-qnn-fp16
python3 download_models.py
sh run.sh /absolute/path/recording.wav --language en-US
```

`download_models.py` fetches the three NPU binaries and the ARM64 native bridge from GitHub Releases and verifies every SHA-256. Python 3.11 or later is required for this downloader. A complete Hugging Face snapshot includes these assets already; use `python3 download_models.py --verify-only` to verify them.

The launcher requires Docker access, the Qualcomm FastRPC/DMA device nodes and the Qualcomm `audio-analytics:1.0.2` container image. It installs **NumPy 1.26.4** into a package-local cache on first use, using network access. Subsequent inference is local; no AI Hub or Hugging Face token is needed. Input must be **16 kHz mono PCM16 WAV**. The container automatically removes itself on exit.

### Repeat measurements

```sh
sh run.sh /absolute/path/recording.wav --warmups 1 --repetitions 5
```

The command prints `TRANSCRIPTION:` records and a detailed `RESULT_JSON:` record, including per-graph HTP timing. For a saved report:

```sh
sh run.sh /absolute/path/recording.wav --output /work/result.json
```

### Inside an existing compatible QNN environment

```sh
python3 -m pip install -r requirements.txt
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu
export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python3 transcribe.py recording.wav --language en-US
```

The included native bridge is Linux ARM64. Its C source and `build_bridge.sh` are provided; rebuilding requires the matching QAIRT SDK headers. Qualcomm libraries and SDK headers are supplied by your runtime/SDK and are not bundled here.

## Model files

| File | Bytes | Function |
|---|---:|---|
| `models/encoder.bin` | 1,200,738,304 | Complete cache-aware FastConformer encoder |
| `models/decoder.bin` | 29,986,816 | Two-layer single-step recurrent prediction decoder |
| `models/joint.bin` | 19,140,608 | Joint network and NPU-side greedy ArgMax |
| **Total model binaries** | **1,249,865,728** | **1.25 GB decimal / 1.16 GiB** |

`manifest.json` contains hashes and immutable release URLs. `config.json`, `languages.json` and `vocab.json` contain the streaming/frontend and tokenizer metadata. `runtime.py` contains only the HTP neural path, the CPU audio frontend, and decoding orchestration.

The original model supports 40 language-locales, with 32 usable out of the box and eight requiring adaptation. **This port's corpus benchmark covers English only.** A locale being present in the prompt dictionary does not establish validated support on this port.

## Benchmark: original NVIDIA model vs this NPU port

Executed on **2026-09-29 on the Arduino**, including inference and scoring. The entire `google/fleurs`, `en_us`, `test` split was evaluated: **647 recordings / 6,387.9 seconds**. No recordings were omitted.

| Metric | Original NVIDIA model card | This FP16 NPU port |
|---|---:|---:|
| Chunk size / language mode | 320 ms / LangID | 320 ms / `en-US` LangID |
| **FLEURS English WER ↓** | **8.27%** | **8.3465%** |
| Difference | — | **+0.0765 percentage points** |
| Full-corpus processing time | Not published for a matching batch-one test | **876.584 s** |
| Real-time factor ↓ | Not published for a matching batch-one test | **0.1372** |
| Processing speed | Different concurrency benchmark | **7.29× real time** |
| Mean / p95 320 ms chunk compute | Not directly comparable | **42.43 / 50.64 ms** |

WER is `1,215 / 14,557` normalized reference words: **875 substitutions, 215 deletions and 125 insertions**. Scoring used `whisper-normalizer==0.1.12` EnglishTextNormalizer symmetrically and `jiwer==4.0.0`; no Whisper neural model was used. NVIDIA describes text normalization but does not specify the exact implementation, so the small score difference cannot be assigned solely to NPU/FP16 arithmetic. With only simple case/punctuation cleanup, our WER is 10.9085%; normalization materially affects the result.

The original card's **7.91% WER uses 1.12-second chunks**, so it is not the matched baseline. Its **960 concurrent streams on an H100 at 320 ms** is a serving-concurrency measurement; our single-stream 7.29× processing factor does not establish seven concurrent streams or a hardware speed ratio.

Every neural invocation produced a positive HTP timing event: **20,245 encoder + 40,855 decoder + 121,188 joint = 182,288 calls**. There were no empty hypotheses or symbol-limit hits. Processing time includes frontend, neural calls, orchestration and text assembly, and excludes audio loading, model initialization, warmups and report writing. The file benchmark does not measure live microphone or final-token latency.

See [`benchmarks/fleurs-en-us.json`](benchmarks/fleurs-en-us.json) for the aggregate measurements and provenance. This release reuses the exact corpus-tested model binary hashes; the standalone runner is refactored from the validated implementation and smoke-tested on the board.

## Conversion

1. Begin with revision-pinned FP16 ONNX exports of the original NVIDIA model. Promote weights to float32 for conversion without recovering precision absent from the FP16 source.
2. Optimize the encoder for QNN, rewriting integer cache-length negation as float negation with casts, exact over its legal 0–56 range.
3. Expand the two single-step LSTMs into supported matrix/gate operations and validate against the original decoder.
4. Remove the joint LogSoftmax (unnecessary for greedy selection) and add NPU-side ArgMax.
5. Compile all graphs for **Arduino VENTUNO Q / V75 / SoC 82** with:

```text
--target_runtime precompiled_qnn_onnx --qairt_version 2.45
--truncate_64bit_io --qnn_options default_graph_htp_precision=FLOAT16
```

This release contains **FP16**, not INT8/W8A16 quantization. All encoder caches and recurrent states are carried across chunks and reset between utterances. Graph contracts are documented in [`MODEL_IO.md`](MODEL_IO.md).

## Sources and licenses

- Original model: [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), **OpenMDW-1.1**; agreement retained in [`LICENSE`](LICENSE).
- Conversion inputs: revision-pinned Soniqo ONNX FP16 export and tokenizer/frontend metadata; exact repositories and revisions in [`provenance.json`](provenance.json).
- Frontend/cache reference: `soniqo/speech-core`, Apache-2.0; attribution in [`NOTICE.md`](NOTICE.md).
- Runner and native bridge source: **Apache-2.0**, [`LICENSE-CODE`](LICENSE-CODE).
- Qualcomm runtime: externally supplied by the referenced container/SDK, under its own terms.

This is a community hardware port, not an official NVIDIA, Qualcomm or Arduino release.

# Attribution and modification notices

## Original model

NVIDIA Nemotron 3.5 ASR Streaming 0.6B.
Source: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
Model license: OpenMDW-1.1, https://openmdw.ai/license/1-1/ (copy in LICENSE).
The original model architecture and learned parameters originate with NVIDIA.

## Intermediate exports and metadata

Soniqo FP16 ONNX export:
https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16
Revision: 76daabfd0aaf5ec6ef1e6640eae3b364af6c9970.

Soniqo vocabulary, prompt dictionary and streaming configuration:
https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-LiteRT-FP16
Revision: 1503a9a1eb75b813b83ba65bf5e9fecea4a46091.

Those export cards cite NVIDIA's Open Model License:
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
Copy retained in `licenses/NVIDIA-Open-Model-License.txt`.
The original NVIDIA model card currently specifies OpenMDW-1.1. The original source and intermediate export notices are retained here without asserting authorship of their model parameters.

## Frontend and streaming implementation

The log-mel frontend and streaming cache wiring follow the Apache-2.0 implementation in soniqo/speech-core:
https://github.com/soniqo/speech-core/tree/f0050757a7d7d24c60b17e60016989ecd07a491d

This port rewrites the runtime integration for native QNN HTP, adds cache-length validation and per-call accelerator profiling, expands the prediction network's LSTM steps into QNN-supported operators, and moves greedy token selection into the joint NPU graph. The standalone runner is refactored from the tested Arduino implementation; neural graphs are unchanged.

## Other components

NumPy is an external dependency and retains its own license. Qualcomm QAIRT libraries and SDK headers are external dependencies, not redistributed in this repository. `libnemotron_qnn_bridge.so` is the compiled version of this repository's native bridge source, not a copy of Qualcomm's runtime.

FLEURS is provided by Google: https://huggingface.co/datasets/google/fleurs. Only aggregate benchmark results are included; corpus audio is not distributed in this release.

# QNN graph I/O contracts

Internal floating-point neural computation is compiled for FP16. Public tensor buffers use float32/int32. Batch size is fixed at one. The bridge reads actual tensor metadata and rejects dtype/shape mismatches. Compiled output names may be `output_N`; the runner maps them by the documented order.

## Encoder — `models/encoder.bin`

| Input | Dtype | Shape |
|---|---|---|
| `audio_signal` | float32 | `[1,128,32]` |
| `audio_length` | int32 | `[1]` |
| `language_mask` | float32 | `[1,128]` |
| `pre_cache` | float32 | `[1,128,9]` |
| `cache_last_channel` | float32 | `[24,1,56,1024]` |
| `cache_last_time` | float32 | `[24,1,1024,8]` |
| `cache_last_channel_len` | int32 | `[1]`, value 0–56 |

Outputs, in order: `encoded_output [1,4,1024]`, `encoded_length [1]`, `new_pre_cache [1,128,9]`, `new_cache_last_channel [24,1,56,1024]`, `new_cache_last_time [24,1,1024,8]`, `new_cache_last_channel_len [1]`. Length tensors are int32; the remaining outputs are float32.

Supply a 32-frame log-mel chunk, a one-hot language mask, and all caches from the preceding step. Start all caches at zero for each utterance. The frontend configuration is recorded in `config.json`.

## Prediction decoder — `models/decoder.bin`

| Input | Dtype | Shape |
|---|---|---|
| `token` | int32 | `[1,1]` |
| `h` | float32 | `[2,1,640]` |
| `c` | float32 | `[2,1,640]` |

Outputs: `decoder_output [1,1,640]`, `h_out [2,1,640]`, `c_out [2,1,640]`, all float32.

Initialize h/c to zero and the token to blank **13087**. Commit the resulting recurrent states only when the joint emits a nonblank token. Cache the prediction output over blank encoder frames.

## Joint + greedy token selection — `models/joint.bin`

| Input | Dtype | Shape |
|---|---|---|
| `encoder_output` | float32 | `[1,1,1024]` |
| `decoder_output` | float32 | `[1,1,640]` |

Output: `token_id`, int32 `[1,1]`, value 0–13087. ArgMax executes inside the NPU graph; full logits are not returned. Blank advances to the next encoder frame; nonblank updates the token/recurrent state and continues the greedy loop.

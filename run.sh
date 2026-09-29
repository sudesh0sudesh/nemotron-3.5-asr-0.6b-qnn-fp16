#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
if [ "$#" -lt 1 ]; then
    printf '%s\n' 'Usage: sh run.sh recording.wav [--language en-US] [--warmups 1 --repetitions 5]' >&2
    exit 2
fi
HERE=$(dirname "$(realpath "$0")")
AUDIO=$(realpath "$1")
shift
test -f "$AUDIO"
test -f "$HERE/models/encoder.bin" || { printf '%s\n' 'Download the model assets first: python3 download_models.py' >&2; exit 1; }
test -c /dev/fastrpc-cdsp-secure
exec docker run --rm --name "nemotron-npu-$$" \
    --label purpose=nemotron-npu-public --memory 8g --pids-limit 256 \
    --device /dev/fastrpc-cdsp --device /dev/fastrpc-cdsp-secure \
    --device /dev/dma_heap/system \
    --mount "type=bind,source=$HERE,target=/work" \
    --mount "type=bind,source=$AUDIO,target=/input.wav,readonly" \
    --mount type=bind,source=/usr/share/qcom,target=/usr/share/qcom,readonly \
    --mount type=bind,source=/etc/ssl/certs/ca-certificates.crt,target=/run/host-ca.pem,readonly \
    --env SSL_CERT_FILE=/run/host-ca.pem --env PIP_CERT=/run/host-ca.pem \
    --env ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp \
    --env LD_LIBRARY_PATH=/usr/lib:/usr/lib/aarch64-linux-gnu \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
    --env PYTHONPATH=/work/.python-runtime --workdir /work \
    --entrypoint /bin/sh \
    artifacts.codelinaro.org/iot-solutions-microservices/audio-analytics:1.0.2 \
    -c '
      set -eu
      if ! python3 -c "import importlib.metadata as m; assert m.version(\"numpy\")==\"1.26.4\"" 2>/dev/null; then
        python3 -m pip install --quiet --no-cache-dir --target /work/.python-runtime -r /work/requirements.txt
      fi
      exec python3 -B -u /work/transcribe.py /input.wav "$@"
    ' sh "$@"

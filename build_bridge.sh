#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT to the matching QAIRT SDK directory}"
HERE=$(dirname "$(realpath "$0")")
case "$(uname -m)" in aarch64|arm64) ;; *) printf '%s\n' 'Build on the ARM64 target with the QAIRT 2.46 SDK headers.' >&2; exit 1;; esac
gcc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared \
  -I "$QNN_SDK_ROOT/include/QNN" "$HERE/nemotron_qnn_bridge.c" \
  -o "$HERE/libnemotron_qnn_bridge.so" -ldl

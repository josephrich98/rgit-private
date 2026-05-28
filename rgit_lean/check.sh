#!/usr/bin/env bash
# Build / check the rgit_lean Lean development.
#
# Works around two issues on RHEL 8 (glibc 2.28):
#   * the bundled Lean toolchain `clang` needs GLIBC_2.29 -> use the system `cc`,
#     with the toolchain's shared libs on the linker / loader path;
#   * the Mathlib cache tool's static `curl` trips over the system OpenSSL
#     config -> neutralize OPENSSL_CONF and point at the system CA bundle.
#
# Usage:
#   ./check.sh          # lake build (kernel-checks every proof)
#   ./check.sh cache    # lake exe cache get (fetch prebuilt Mathlib oleans)
set -euo pipefail

TC="$(lean --print-prefix 2>/dev/null || echo "$HOME/.elan/toolchains/leanprover--lean4---v4.30.0")"

export LEAN_CC=/usr/bin/cc
export LIBRARY_PATH="$TC/lib:$TC/lib/lean4${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TC/lib:$TC/lib/lean4${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENSSL_CONF=/dev/null
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

cd "$(dirname "$0")"

case "${1:-build}" in
  cache) exec lake exe cache get ;;
  build) exec lake build ;;
  *)     exec lake "$@" ;;
esac

#!/usr/bin/env bash
# Create a locally trusted TLS certificate so phones on the LAN can open the
# client over HTTPS (browsers only allow camera access in a secure context).
#
# Needs mkcert: `brew install mkcert` (macOS) or see https://github.com/FiloSottile/mkcert.
# Phones must trust the mkcert root CA: install "$(mkcert -CAROOT)/rootCA.pem"
# on the phone (iOS: AirDrop it, install the profile, then enable full trust in
# Settings > General > About > Certificate Trust Settings).
set -euo pipefail

out_dir="data/certs"
mkdir -p "$out_dir"

if ! command -v mkcert >/dev/null 2>&1; then
  echo "mkcert not found. Install it first: brew install mkcert" >&2
  exit 1
fi

lan_ip="${1:-}"
if [[ -z "$lan_ip" ]]; then
  lan_ip="$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -z "$lan_ip" ]]; then
  echo "Could not detect LAN IP. Pass it: $0 192.168.1.20" >&2
  exit 1
fi

mkcert -install
mkcert -cert-file "$out_dir/cert.pem" -key-file "$out_dir/key.pem" \
  "$lan_ip" localhost 127.0.0.1 ::1

cat <<MSG
Certificate written to $out_dir/. Add to config/.env:

  POCKETSLAM_HOST=0.0.0.0
  POCKETSLAM_TLS_CERT=$out_dir/cert.pem
  POCKETSLAM_TLS_KEY=$out_dir/key.pem

Then open https://$lan_ip:8443 on the phone.
MSG

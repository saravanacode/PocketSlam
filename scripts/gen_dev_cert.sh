#!/usr/bin/env bash
# Create a TLS certificate so phones on the LAN can open the capture page over
# HTTPS (browsers only allow camera and motion sensors in a secure context).
#
# With mkcert (`brew install mkcert`): the certificate is trusted once you
# install "$(mkcert -CAROOT)/rootCA.pem" on the phone
# (Android: Settings > Security > Encryption & credentials > Install a certificate > CA certificate).
#
# Without mkcert: a self-signed certificate is made with openssl. Chrome on
# Android shows a warning the first time; tap Advanced > Proceed. Camera and
# sensors work after that.
set -euo pipefail

out_dir="data/certs"
mkdir -p "$out_dir"

lan_ip="${1:-}"
if [[ -z "$lan_ip" ]]; then
  lan_ip="$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
if [[ -z "$lan_ip" ]]; then
  echo "Could not detect the LAN IP. Pass it: $0 192.168.1.20" >&2
  exit 1
fi

if command -v mkcert >/dev/null 2>&1; then
  mkcert -install
  mkcert -cert-file "$out_dir/cert.pem" -key-file "$out_dir/key.pem" \
    "$lan_ip" localhost 127.0.0.1 ::1
  trust_note="Install \"$(mkcert -CAROOT)/rootCA.pem\" on the phone once to avoid warnings."
else
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$out_dir/key.pem" -out "$out_dir/cert.pem" \
    -subj "/CN=PocketSlam dev" \
    -addext "subjectAltName=IP:$lan_ip,IP:127.0.0.1,DNS:localhost" 2>/dev/null
  trust_note="Self-signed: on the phone tap Advanced > Proceed the first time."
fi
chmod 600 "$out_dir/key.pem"

cat <<MSG
Certificate for $lan_ip written to $out_dir/.
$trust_note

Add to config/.env:

  POCKETSLAM_TLS_CERT=$out_dir/cert.pem
  POCKETSLAM_TLS_KEY=$out_dir/key.pem

Then run: uv run python -m pocketslam
Phone:     https://$lan_ip:10001
Dashboard: http://127.0.0.1:8000 (8001 if 8000 is busy)
MSG

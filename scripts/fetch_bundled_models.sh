#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#
# Download the nano (n) YOLO26 Android models into the example app's asset folder at build time so the app ships with
# them by default. The models are NOT committed to git (they are large binaries, gitignored via *.tflite);
# this script fetches them on demand during the platform build.
#
# YOLOModelResolver checks assets/models/ before falling back to a network download, so a bundled model means no
# first-run download for the user.
#
# Best-effort: if a download fails (e.g. offline build machine), the script warns and exits 0 so the build still
# succeeds — the app simply falls back to the existing runtime download for any missing model. Bundling is skipped
# entirely under CI (CI / GITHUB_ACTIONS) to keep GitHub builds fast and off the network; set FORCE_BUNDLED_MODELS=1
# to override.
#
# Usage: fetch_bundled_models.sh
#

set -u

# Skip bundling under CI.
if [ "${FORCE_BUNDLED_MODELS:-}" != "1" ] && { [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; }; then
  echo "fetch_bundled_models: CI detected (CI/GITHUB_ACTIONS set); skipping model bundling — app will download at runtime."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$REPO_ROOT/example/assets/models"

# Android release source.
BASE="https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6"

# Only the models currently exposed by the app.
FILES=(
  "yolo26n_w8a32.tflite"
  "yolo26n-seg_w8a32.tflite"
  "yolo26n-pose_w8a32.tflite"
)

mkdir -p "$DEST"

fetch() {
  # Download $1 from $BASE into $DEST, atomically, skipping if already present and non-empty.
  local name="$1"
  local out="$DEST/$name"

  if [ -s "$out" ]; then
    echo "fetch_bundled_models: have $name"
    return 0
  fi

  local tmp="$out.download"
  rm -f "$tmp"

  echo "fetch_bundled_models: downloading $name"

  if curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 -o "$tmp" "$BASE/$name"; then
    if [ ! -s "$tmp" ]; then
      echo "fetch_bundled_models: WARNING $name downloaded 0 bytes; will fall back to runtime download" >&2
      rm -f "$tmp"
    else
      mv -f "$tmp" "$out"
    fi
  else
    echo "fetch_bundled_models: WARNING failed to download $name; will fall back to runtime download" >&2
    rm -f "$tmp"
  fi
}

for f in "${FILES[@]}"; do
  fetch "$f"
done

# Always succeed: bundling is an optimization, never a hard build dependency.
exit 0
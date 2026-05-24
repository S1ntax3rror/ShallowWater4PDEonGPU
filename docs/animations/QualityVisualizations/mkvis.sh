#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
fps=12
width=960

shopt -s nullglob

find "$input_dir" -maxdepth 1 -type f -name '*.mp4' -print0 |
while IFS= read -r -d '' path; do
	gif="${path%.mp4}.gif"
	palette="$(mktemp --suffix=.png)"

	ffmpeg -nostdin -y -i "$path" -vf "fps=${fps},scale=${width}:-1:flags=lanczos,palettegen" "$palette"
	ffmpeg -nostdin -y -i "$path" -i "$palette" -lavfi "fps=${fps},scale=${width}:-1:flags=lanczos[x];[x][1:v]paletteuse" "$gif"

	rm -f "$palette"
done


#!/usr/bin/env bash
set -euo pipefail

YESQL_SRC="/home/YeSQL/"

if [ ! -d "${YESQL_SRC}" ]; then
    echo "YeSQL source directory not found at ${YESQL_SRC}" >&2
    exit 1
fi

copy_to_container() {
    local src="$1"
    local dest_container="$2"
    local dest_path="$3"
    docker exec -it ${dest_container} rm -rf /home/YeSQL/
    docker exec -it ${dest_container} ls
    echo "Copying ${src} -> ${dest_container}:${dest_path}"
    docker cp "${src}" "${dest_container}:${dest_path}"
}

copy_to_container "${YESQL_SRC}" "monetdb-localworker2" "/home/"


run_in_background() {
    local container="$1"
    echo "Starting run_sample.sh inside ${container} (background)."
    docker exec "${container}" bash -c "cd YeSQL && python3 -m pip install cffi && bash exec.sh && echo 'done!!!'" &
    BG_PIDS+=("$!")
}

BG_PIDS=()
run_in_background "monetdb-localworker2"

echo "Background PIDs: ${BG_PIDS[*]}"
echo "Done dispatching commands."

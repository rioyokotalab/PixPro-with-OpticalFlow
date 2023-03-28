#!/bin/bash
script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
bdd100k_root=$(
    cd "$1" || exit
    pwd
)

pushd "$bdd100k_root"

# mkdir -p "$script_dir"/unzip_log
# mkext_name=$(mktemp "$script_dir"/unzip_log/XXX.log)
# find "$bdd100k_root" -maxdepth 1 -name 'bdd100k_videos_*.zip' | parallel --joblog "$mkext_name" -u unzip -o
find "$bdd100k_root" -maxdepth 1 -name 'bdd100k_videos_*.zip' | parallel --eta --progress -u unzip -o

popd

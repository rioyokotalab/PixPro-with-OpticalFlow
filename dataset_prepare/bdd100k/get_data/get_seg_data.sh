#!/bin/bash
set -eu
script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
bdd100k_root=$(
    cd "$1" || exit
    pwd
)

sem_seg_root_path="$bdd100k_root/sem_seg_root"
data_root="$bdd100k_root/bdd100k"
seg_data_root="$data_root/seg"
mkdir -p "$seg_data_root"

pushd "$sem_seg_root_path"
zips=$(find "$sem_seg_root_path" -maxdepth 1 -name "bdd100k_*.zip")
for zip_file in ${zips};
do
    unzip -o "$zip_file"
done
pushd "$seg_data_root"
cp -r "$sem_seg_root_path/bdd100k/images/10k" "images"
cp -r "$sem_seg_root_path/bdd100k/labels/sem_seg/colormaps" "labels"
popd
popd

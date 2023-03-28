#!/bin/bash
script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
bdd100k_root=$(
    cd "$1" || exit
    pwd
)

if type "aria2c" > /dev/null 2>&1
then
    aria2c -x16 -s16 --auto-file-renaming=false -i "$script_dir"/input.txt
else
    wget -i "$script_dir"/input.txt
fi

log_root_dir="$script_dir"/process_log
mkdir -p "$log_root_dir"
log_dir=$(mktemp -d "$log_root_dir"/processXXX)

mkext_name="$log_dir/unzip.log"
find "$bdd100k_root"  -maxdepth 1 -name 'bdd100k_videos_*.zip' | parallel --joblog "$mkext_name" -u unzip -n

# データの置き場所に応じて変える
dirPath="$bdd100k_root/bdd100k/videos"
resPath="$bdd100k_root/bdd100k/images"

subsets=`find ${dirPath}/* -maxdepth 0 -type d | xargs basename -a`

mkdir -p ${resPath}

for subset in ${subsets};
do
    echo start processing ${subset}...
    mkdir -p ${resPath}/${subset}
    mkjob_name="$log_dir/mkdir_"$subset".log"
    ffmjob_name="$log_dir/ffmpeg_"$subset".log"
    find ${dirPath}/${subset}/ -maxdepth 1 -type f -name "*.mov" | parallel --joblog "$mkjob_name" mkdir -p ${resPath}/${subset}/{/.}
    find ${dirPath}/${subset}/ -maxdepth 1 -type f -name "*.mov" | parallel --joblog "$ffmjob_name" --eta --progress ffmpeg -i {} -vcodec mjpeg -r 10 -an -q:v 0 -f image2  ${resPath}/${subset}/{/.}/%05d.jpg
done

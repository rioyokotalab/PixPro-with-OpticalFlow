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

# データの置き場所に応じて変える
dirPath="$bdd100k_root/bdd100k/videos"
resPath="$bdd100k_root/bdd100k/images"

mkdir -p ${resPath}

order_s="$2"
order_e="$3"
subset=${4:-"train"}
job_num=8
# job_num=2
part_num=$((($3 + $job_num - 1) / $job_num))
# tmpdir="$script_dir"/log_train/"$order_s"-"$order_e"

# mkdir -p "$tmpdir"
# logdir=$(mktemp -d "$tmpdir"/XXX)

echo start processing ${subset}...
mkdir -p ${resPath}/${subset}
# ffmjob_name="$logdir"/ffmpegjob_"$subset".log
mov_all_list=$(find ${dirPath}/${subset}/ -maxdepth 1 -type f -name "*.mov" | sort | tail -n +"$2" | head -n "$3")
# echo "$mov_all_list"
for i in $(seq $part_num);
do
    mov_list=$(echo "$mov_all_list" | tail -n +"$((($i - 1) * $job_num + 1))" | head -n $job_num)
    echo "$i"
    echo "$mov_list"
    # echo "$mov_list" | parallel --joblog "$ffmjob_name" --eta --progress ffmpeg -i {} -vcodec mjpeg -r 10 -an -q:v 0 -f image2 ${resPath}/${subset}/{/.}/%05d.jpg
    echo "$mov_list" | parallel -j $job_num --eta --progress ffmpeg -i {} -vcodec mjpeg -r 10 -an -q:v 0 -f image2 ${resPath}/${subset}/{/.}/%05d.jpg
done

popd

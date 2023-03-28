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


if type "aria2c" > /dev/null 2>&1
then
    aria2c -x16 -s16 --auto-file-renaming=false -i "$script_dir"/input.txt
else
    wget -nc -i "$script_dir"/input.txt
fi

popd

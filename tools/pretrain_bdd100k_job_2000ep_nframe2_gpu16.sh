#!/bin/bash

#------ pjsub option --------#
#PJM -g jhpcn2628
#PJM -L node=4
#PJM --mpi proc=16
#PJM -L rscgrp=cxgfs-small
#PJM -L elapse=168:00:00
#PJM -N nframe2_2000ep_bdd100k_pretrain_pixpro
#PJM -j
#PJM -X

set -x

echo "start scirpt file1 cat"
cat "$0"

set +x

echo "end scirpt file1 cat"

START_TIMESTAMP=$(date '+%s')

# ======== Args ========

bs=${BS:-64}
pixpro_pos_ratio=${POS_RATIO:-0.7}
# for raft
alpha1=${ALPHA1:-0.01}
alpha2=${ALPHA2:-0.5}
flow_bs=${FLOW_BS:-2}
is_use_flow_frames=${FLOW_FRAMES:-"n"}
flow_up=${FLOW_UP:-"y"}

cur_rel=${CUR_REL:-"n"}

base_script=${SCRIPT_NAME:-"./pretrain_bdd100k_job_base.sh"}

# ======== Variables ========

## read only vals
export ALL_EPOCH=2000
export DEBUG="n"
export EPOCH=10
export N_FRAME=2
export OPT_LEV="O0"
# for raft
export USE_FLOW="y"
export RAFT_NAME="small"
export USE_MASK="y"
export FLOW_FILES="y"

## changeable vals
export BS=$bs
export POS_RATIO=$pixpro_pos_ratio

# for raft
export ALPHA1=$alpha1
export ALPHA2=$alpha2
export FLOW_BS=$flow_bs
export FLOW_FRAMES="$is_use_flow_frames"
export FLOW_UP="$flow_up"

export CUR_REL="$cur_rel"


# ======== Scripts ========

bash "$base_script"

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"


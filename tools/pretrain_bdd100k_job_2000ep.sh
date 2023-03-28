#!/bin/bash

#------ pjsub option --------#
#PJM -g jhpcn2628
#PJM -L node=2
#PJM --mpi proc=8
#PJM -L rscgrp=cxgfs-small
#PJM -L elapse=168:00:00
#PJM -N no_raft_2000ep_bdd100k_pretrain_pixpro
#PJM -j
#PJM -X

set -x

echo "start scirpt file1 cat"
cat "$0"

set +x

echo "end scirpt file1 cat"

START_TIMESTAMP=$(date '+%s')

# ======== Args ========

bs=${BS:-128}
pixpro_pos_ratio=${POS_RATIO:-0.7}
n_frame=${N_FRAME:-1}
opt_level=${OPT_LEV:-"O1"}

cur_rel=${CUR_REL:-"n"}

base_script=${SCRIPT_NAME:-"./pretrain_bdd100k_job_base.sh"}

# ======== Variables ========

export ALL_EPOCH=2000
export DEBUG="n"
export EPOCH=10
# for raft
export USE_FLOW="n"

export BS=$bs
export POS_RATIO=$pixpro_pos_ratio
export N_FRAME=$n_frame
export OPT_LEV="$opt_level"

export CUR_REL="$cur_rel"


# ======== Scripts ========

bash "$base_script"

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"


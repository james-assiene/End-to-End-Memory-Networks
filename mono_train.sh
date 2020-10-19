#!/bin/bash

optimizer=adam
learning_rate=1e-2
batch_size=2000
weight_tying="layer-wise"
num_memory_hops=3
temporal_encoding="yes"
edim=100
lr_scheduler=reduceonplateau
patience=30
output_dir="/scratch/jassiene/E2EMemNNExperiments"
datapath=/scratch/jassiene/data

model_name=m_${optimizer}_lr_${learning_rate}_bs_${batch_size}_wt_${weight_tying}_nmh_${num_memory_hops}_tmpe_${temporal_encoding}_edim_${edim}_lrscheduler_${lr_scheduler}
rm -fr $output_dir/*

export PYTHONPATH=$PYTHONPATH:$HOME
agent_path=~/parlai_internal/agents/end2end_mem_nn/
rm -fr $agent_path
mkdir -p $agent_path
touch ~/parlai_internal/__init__.py
touch ~/parlai_internal/agents/__init__.py
cp -R ./*.py $agent_path
#:task1k:1
python -m parlai.scripts.train_model --datapath $datapath --tensorboard_log=True  --tensorboard-log True --train-predict True -stim 120 -m internal:end2end_mem_nn -t babi \
     -bs $batch_size -veps 3 -mf "${output_dir}/$model_name" -nmh $num_memory_hops -wt $weight_tying -tmpe $temporal_encoding --embedding-dim $edim \
     --optimizer $optimizer --lr_scheduler $lr_scheduler --learningrate $learning_rate -vp $patience

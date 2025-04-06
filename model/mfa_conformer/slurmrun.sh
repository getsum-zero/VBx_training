#!/usr/bin/env bash

cmd="/home3/yihao/slurm.pl --quiet" #  --nodelist=node06 --gpu 1

conda init
source ~/.bashrc
conda activate mfa

$cmd log/mfa.log \
python3 inference.py --checkpoint_path="/home3/yihao/Research/Code/VBx/VBx-sep-recluster/pre_trained/mfa/epoch=17_cosine_eer=1.24.ckpt"  --audio_path="/home3/yihao/Research/Code/VBx/VBx-sep-recluster/model/mfa_conformer_sv/audio_samples/DH_EVAL_0175_12.637_14.780_spk0.wav"
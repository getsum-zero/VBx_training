
#!/bin/bash
# -*- coding: utf-8 -*-

set -eu pipefail

stage_getlist=0
stage_mix=0
stage_embedding=0
embedding_type=mfa_conformer
stage_plda=1


dataset=/home/getsum/code/speech/data/LibriSpeech/test-clean
noiseset=/home/getsum/code/speech/data/STRAFFIC
workspace=/home/getsum/code/speech/VBx_training
expfolder=$workspace/data/libirspeech_musan

if [ $stage_getlist -eq 1 ]; then

    # wav
    # voxceleb1 use .wav; voxceleb2 use .m4a
    cd $workspace
    python utils/getlist_voxceleb.py --wavs_dir $dataset \
                --out_dir $expfolder \
                --file_type .flac
    
    # noise
    python utils/getlist_noise.py --wavs_dir $noiseset \
                --out_dir $expfolder \
                --file_type .wav
fi


if [ $stage_mix -eq 1 ]; then
    cd $workspace
    mixture_dir=/home/getsum/code/speech/data/mixture
    echo 'mixing speech and noise'
    python utils/mix_speech_noise.py --speech_list $expfolder/data.list \
                --noise_list $expfolder/noise.list \
                --out_dir $expfolder \
                --save_dir $mixture_dir 
fi

if [ $stage_embedding -eq 1 ]; then
    cd $workspace

    if [ $embedding_type == "mfa_conformer" ]; then
        export KALDI_ROOT=../  # 随便设置避免出现warning

        # mfa_conformer
        echo "get embedding via mfa_conformer"
        save_dir=/home/getsum/code/speech/data/embedding
        python -m embedding.mfa_conformer --scp_file $expfolder/mixture.list \
                --out_dir $expfolder \
                --save_dir $save_dir \
                --checkpoint_path pretrained/MFA_conformer/epoch=17_cosine_eer=0.72.ckpt
    fi 
fi

if [ $stage_plda -eq 1 ]; then
    cd $workspace
    echo "prepare spk2utt utt2spk embedding.txt"
    python utils/list2utt.py --scp_file $expfolder/embedding.list \
                --out_dir $expfolder \
    
    echo "training plda & transform.h5"
    cd $workspace/plda_recipe

    cat $expfolder/output_vectors.txt | python local/train_transform.py --utt2spk $expfolder/utt2spk --output-h5 $expfolder/transform.h5 
    echo "transform.h5 Generated"
    cat $expfolder/output_vectors.txt | ivector-compute-plda ark:$expfolder/spk2utt ark,cs:- $expfolder/plda
    echo "plda Generated"
fi


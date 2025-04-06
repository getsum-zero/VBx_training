#!/bin/bash

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

__conda_setup="$('/home3/yihao/anaconda3' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home3/yihao/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home3/yihao/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home3/yihao/anaconda3/bin:$PATH"
    fi
fi

unset __conda_setup
echo "conda init"

nnet_dir=exp/xvector_nnet
stage=7
train_stage=-1

. ./cmd.sh || exit 1
. ./path.sh || exit 1
set -e
. ./utils/parse_options.sh



if [ ${stage} -le 7 ]; then
    # train PLDA
    # for i in $(seq 0 $((num_gpus-1)))
    # do
    #     cat exp/xvectors_plda_train_$((i+1))/*.txt
    # done | python local/train_transform.py --utt2spk data/plda_train/utt2spk --output-h5 exp/transform.h5 | ivector-compute-plda ark:data/plda_train/spk2utt ark,cs:- exp/plda
    conda activate VBx 
    export KALDI_ROOT=../  # 随便设置避免出现warning
    # 定义文件所在的根目录
    root_dir="/home3/yihao/Research/Code/kaldi/egs/sre16/VBx-training-recipe/mfa_conformer_sv/output3/"
    output_utt2spk="data/plda_train/utt2spk"
    output_spk2utt="data/plda_train/spk2utt"
    output_vectors="data/plda_train/output_vectors.txt"

    # 检查文件是否存在，若存在则删除
    for file in "$output_utt2spk" "$output_spk2utt" "$output_vectors"; do
        if [ -f "$file" ]; then
            echo "File $file exists. Deleting it..."
            rm "$file"
        fi
    done

    # 遍历根目录下所有txt文件
    find "$root_dir" -type f -name "*.txt" | while read file; do
        # 获取文件路径中的 id 和子文件夹名
        path_parts=($(echo $file | tr "/" "\n"))
        id="${path_parts[-3]}"  # id10003
        folder="${path_parts[-2]}"  # _JpHD6VnJ3I
        filename="${path_parts[-1]}"  # 00001.txt

        # 去掉文件名的扩展名 ".txt"
        utt_id="${filename%.*}"

        # 输出到文件
        echo "$id"_"$folder"_"$utt_id $id" >> "$output_utt2spk"

        # 读取整个文件内容，并转换成一行
        line=$(tr -d '\n' < "$file")
        #echo "原始文件内容：$line"

        # 去掉外层的中括号"[]" 针对celeb1
        #processed_line=$(echo "$line" | awk '{sub(/\[\s*\[/, "["); sub(/\]\s*\]/, "]"); print}')
        #echo "去掉外层中括号后的内容：$processed_line"

        # 将转换后的一行数据写入 output_vectors.txt
        echo "$line" >> "$output_vectors"
    done
    echo "output_vectors.txt and output_utt2spk Generated: $output_vectors , $output_utt2spk"
    # 创建 spk2utt 文件
    awk '
    {
        spk=$2
        utt=$1
        spk_to_utt[spk] = spk_to_utt[spk] " " utt
    }
    END {
        for (spk in spk_to_utt) {
            print spk spk_to_utt[spk]
        }
    }' "$output_utt2spk" > "$output_spk2utt"

    # echo "spk2utt Generated: $output_spk2utt"

    cat data/plda_train/output_vectors.txt | python local/train_transform.py --utt2spk data/plda_train/utt2spk --output-h5 exp/transform.h5 
    # echo "transform.h5 Generated"

    cat data/plda_train/output_vectors.txt | ivector-compute-plda ark:data/plda_train/spk2utt ark,cs:- exp/plda
    # echo "plda Generated"

fi


exit 0

# -*- coding: utf-8 -*-
# @Date     : 2025/04/03
# @Author   : Getsum
# @File     : getlist_voxceleb.py
# @Github   : https://github.com/getsum-zero

# @Description:
# 1. This script is only applicable to the voxceleb dataset
# 2. This script is used to generate the json list files for training and testing.
# 3. It reads the audio files and their corresponding speaker from the specified directories.

from argparse import ArgumentParser
import os
import glob
import json
import soundfile as sf

parser = ArgumentParser(add_help=True)
parser.add_argument('--wavs_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--file_type', type=str, default='.wav')
args = parser.parse_args()

if __name__ == '__main__':

    os.makedirs(args.out_dir, exist_ok=True)
    wavs = glob.glob(os.path.join(args.wavs_dir, f'**/*{args.file_type}'), recursive=True)

    # Iterate through each .wav file
    # voxceleb1:  $path/id10003/na8-QEFmj44/00003.wav
    # 
    outfile = os.path.join(args.out_dir, 'data.list')
    with open(outfile, 'w') as f:
        for wav in wavs:
            splits = wav.split('/')
            key = "_".join(splits[-3:]).replace(args.file_type, '')
            dur = sf.info(wav).duration
            dic = {
                'key': key,
                'wav': os.path.abspath(wav),
                'speaker_id': splits[-3],
                'duration': dur
            }
            f.write(json.dumps(dic) + '\n')

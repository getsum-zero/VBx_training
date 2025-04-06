# -*- coding: utf-8 -*-
# @Date     : 2025/04/03
# @Author   : Getsum
# @File     : getlist_noise.py.py
# @Github   : https://github.com/getsum-zero

# @Description:


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


    outfile = os.path.join(args.out_dir, 'noise.list')
    with open(outfile, 'w') as f:
        for wav in wavs:
            splits = wav.split('/')
            key = splits[-1].replace(args.file_type, '')
            dur = sf.info(wav).duration
            dic = {
                'key': key,
                'wav': os.path.abspath(wav),
                'env': "unknow",
                'duration': dur
            }
            f.write(json.dumps(dic) + '\n')

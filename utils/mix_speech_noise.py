# -*- coding: utf-8 -*-
# @Date     : 2025/04/03
# @Author   : Getsum
# @File     : mix_speech_noise.py
# @Github   : https://github.com/getsum-zero

# @Description: Mix speech and noise
# ref: https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriMix/separation/dynamic_mixing.py

import os
import random
import numpy as np
import soundfile as sf
from argparse import ArgumentParser
import json
import pyloudnorm
from tqdm import tqdm
import warnings

parser = ArgumentParser(add_help=True)
parser.add_argument('--speech_list', type=str)
parser.add_argument('--noise_list', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--save_dir', type=str, default=None)

parser.add_argument('--min_loudness', type=int, default=-33)
parser.add_argument('--max_loudness', type=int, default=-25)
parser.add_argument('--max_amp', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--sample_rate', type=int, default=16000)

args = parser.parse_args()
random.seed(args.seed)



def read_list(path):
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list



def normalize(meter, signal, is_noise=False):
    """
    This function normalizes the audio signals for loudness
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(signal)
        if is_noise:
            target_loudness = random.uniform(
                args.min_loudness - 5, args.max_loudness - 5
            )
        else:
            target_loudness = random.uniform(args.min_loudness, args.max_loudness)
        signal = pyloudnorm.normalize.loudness(
            signal, c_loudness, target_loudness
        )

        # check for clipping
        if np.max(np.abs(signal)) >= 1:
            signal = signal * args.max_amp / np.max(np.abs(signal))
        return signal


def compute_gain(meter, waveform, waveform_norm):
    """ Compute the gain between the original and target loudness"""
    loudness = meter.integrated_loudness(waveform)
    renormalize_loudness = meter.integrated_loudness(waveform_norm)
    delta_loudness = renormalize_loudness - loudness
    gain = np.power(10.0, delta_loudness / 20.0)
    return gain



def mix_loudness(meter, speech, noise):
    speech_norm = normalize(meter, speech)
    noise_norm = normalize(meter, noise, is_noise=True)
    mixture = speech_norm + noise_norm

    max_amp_insig = np.max(np.abs(mixture))
    if max_amp_insig > args.max_amp:
        weight = args.max_amp / max_amp_insig
    else:
        weight = 1
    mixture = weight * mixture
    speech_gain = compute_gain(meter, speech, weight * speech_norm)
    noise_gain = compute_gain(meter, noise, weight * noise_norm)
    snr = 10 * np.log10(np.mean((weight * speech_norm)**2) / np.mean((weight * noise_norm)**2))

    return mixture, [speech_gain, noise_gain], snr
        


if __name__ == '__main__':

    speech_list = read_list(args.speech_list)
    noise_list = read_list(args.noise_list) 
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_dir :
        os.makedirs(args.save_dir, exist_ok=True)

    meter = pyloudnorm.Meter(args.sample_rate)
    mixture_list = os.path.join(args.out_dir, 'mixture.list')

    # mix
    # 混合可以使用多种方案：
    #     y 1. 随机选择一个噪声文件和一个语音文件进行混合; 长噪声截取，短噪声循环拼接 
    #     n 2. 选取一个长于x语音的噪声文件，随机截取一段进行混合 
    # 混合语音数目
    #     y 1. 和语音文件数目相同  
    # 如何混合：
    #     y 1. 相对响度，增益

    with open(mixture_list, 'w') as f:
        for speech in tqdm(speech_list):
            noise = random.choice(noise_list)
            
            speech_wavform, speech_sr = sf.read(speech['wav'])
            noise_wavform, noise_sr = sf.read(noise['wav'])
            assert speech_sr == noise_sr, f"Sampling rate mismatch: {speech_sr} vs {noise_sr}"
            assert speech_sr == args.sample_rate, f"Sampling rate mismatch: {speech_sr} vs {args.sample_rate}"

            if len(speech_wavform.shape) > 1:
                speech_wavform = speech_wavform[:, 0]
            if len(noise_wavform.shape) > 1:
                noise_wavform = noise_wavform[:, 0]

            # Align speech and noise lengths
            len_speech = len(speech_wavform)
            len_noise = len(noise_wavform)
            if len_speech > len_noise:
                repeat_times = len_speech  // len_noise + 1          
                extended_data = np.tile(noise_wavform, repeat_times)
                noise_wavform = extended_data[:len_speech]
                align_type = 'extend'
            else: 
                st = np.random.randint(0, len_noise - len_speech)
                noise_wavform = noise_wavform[st:st+len_speech]
                align_type = f'cut:{st}-{st+len_speech}'
            
            # mix speech and noise
            #     y  1. loudness + gain
            #     n  2. snr
            mixture, gain, snr = mix_loudness(meter, speech_wavform, noise_wavform)
            mixkey = speech['key'] + '_' + noise['key']
            mixture_dic = {
                'key': mixkey,
                'snr': snr,
                'speech_gain': gain[0],
                'noise_gain': gain[1],
                'align_type': align_type,
                'speech': speech,
                'noise': noise,
            }
            if args.save_dir:
                save_path = os.path.join(args.save_dir, mixkey + '.wav')
                output = (mixture * 32768).astype(np.int16).tobytes()
                sf.write(save_path, np.frombuffer(output, dtype=np.int16), args.sample_rate)
                mixture_dic.update({'wav': save_path,})
            f.write(json.dumps(mixture_dic) + '\n')





    
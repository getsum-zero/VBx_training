import os

from tqdm import tqdm
import json
from argparse import ArgumentParser


parser = ArgumentParser(add_help=True)
parser.add_argument("--scp_file", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
args = parser.parse_args()

def iter_speaker_ids(data, target_key="speaker_id"):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                yield value
            yield from iter_speaker_ids(value, target_key)
    elif isinstance(data, list):
        for item in data:
            yield from iter_speaker_ids(item, target_key)

if __name__ == "__main__":
    os.makedirs(args.out_dir, exist_ok=True)
    utt2spk = os.path.join(args.out_dir, 'utt2spk')
    spk2utt = os.path.join(args.out_dir, 'spk2utt')
    output_vectors = os.path.join(args.out_dir, 'output_vectors.txt')

    dict_speaker = {}
    with open(args.scp_file, 'r') as f, \
            open(utt2spk, 'w') as utt2spk_f, open(output_vectors, 'w') as output_vectors_f:
        for line in f:
            data = json.loads(line.strip())
            key = data['key']
            speaker_ids = list(iter_speaker_ids(data))
            assert len(speaker_ids) == 1, f"Multiple speaker IDs found: {speaker_ids}"
            speaker_ids = speaker_ids[0]

            utt2spk_f.write(f"{key} {speaker_ids}\n")
            with open(data['embedding'], 'r') as embedding_f:
                lines = embedding_f.readlines()
                output_vectors_f.write(" ".join([line.strip() for line in lines]) + '\n')
            
            if dict_speaker.get(speaker_ids) is None:
                dict_speaker[speaker_ids] = [key]
            else:
                dict_speaker[speaker_ids].append(key)
            
            
    with open(spk2utt, 'w') as spk2utt_f:
        for key, value in dict_speaker.items():
            spk2utt_f.write(f"{key} {' '.join(value)}\n")
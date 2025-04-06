from model.mfa_conformer.main import Task
from model.mfa_conformer.module.dataset import load_audio

import torch
import os

from tqdm import tqdm
import json
from argparse import ArgumentParser


parser = ArgumentParser(add_help=True)
parser.add_argument("--scp_file", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)

parser.add_argument("--checkpoint_path", type=str, default='../pretrained/MFA_conformer/epoch=17_cosine_eer=0.72.ckpt')

parser.add_argument('--embedding_dim', type=int, default=192)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--second', type=int, default=1.44) 
parser.add_argument("--loss_name", type=str, default="amsoftmax")
parser.add_argument("--input_layer", type=str, default="conv2d2")
parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")
parser.add_argument("--threshold", type=int, default=0.22)
args = parser.parse_args()



def load_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lightning_model = Task(
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        loss_name=args.loss_name,
        input_layer=args.input_layer,
        pos_enc_layer_type=args.pos_enc_layer_type
    )
    lightning_model.eval()
    state_dict = torch.load(args.checkpoint_path, map_location=device)["state_dict"]
    lightning_model.load_state_dict(state_dict)
    lightning_model.to(device)
    return lightning_model, device

def write_txt_vectors(path, data_dict):
    """ Write vectors file in text format.(from vbx training)

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict):
            vector = data_dict[name].flatten()  # 确保是 1D 数组
            f.write('{}  [ {} ]\n'.format(name, ' '.join(map(str, vector))))




def main():

    lightning_model, device = load_model(args)

    with open(args.scp_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    embedding_list = os.path.join(args.out_dir, "embedding.list")

    with open(args.scp_file, 'r') as f, open(embedding_list, 'w') as out_f:
        for line in tqdm(f, total=total_lines, desc="Processing files"):
            data = json.loads(line.strip())
            wav = data['wav']
            key = data['key']

            wav = load_audio(wav, args.second)
            emb_wav = lightning_model(torch.FloatTensor(wav).unsqueeze(0).to(device))
            emb_wav = emb_wav.detach().cpu().numpy()

            emb_dict = {key: emb_wav}
            out_path = os.path.join(args.save_dir, key + ".txt")
            write_txt_vectors(out_path, emb_dict)
            data.update({"embedding": out_path})
            out_f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    main()
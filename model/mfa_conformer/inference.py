from main import Task
from argparse import ArgumentParser
from module.dataset import load_audio
import torch
import numpy as np

parser = ArgumentParser(add_help=True)
parser.add_argument('--embedding_dim', type=int, default=192)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--second', type=int, default=3)
parser.add_argument("--loss_name", type=str, default="amsoftmax")
parser.add_argument("--input_layer", type=str, default="conv2d2")
parser.add_argument("--pos_enc_layer_type", type=str, default="abs_pos")
parser.add_argument("--checkpoint_path", type=str, default='checkpoints/epoch=17_cosine_eer=1.24.ckpt')
parser.add_argument("--audio_path", type=str)
parser.add_argument("--save_path", type=str)

hparams = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading model
lightning_model = Task(
                    embedding_dim=hparams.embedding_dim,
                    num_blocks=hparams.num_blocks,
                    loss_name=hparams.loss_name,
                    input_layer=hparams.input_layer,
                    pos_enc_layer_type=hparams.pos_enc_layer_type
                )
lightning_model.eval()
state_dict = torch.load(hparams.checkpoint_path, map_location=device)["state_dict"]
lightning_model.load_state_dict(state_dict)
lightning_model.to(device)
#print("load weight from {}".format(hparams.checkpoint_path))

# Running inference
audio_path = hparams.audio_path
wav = load_audio(audio_path, hparams.second)
emb_wav = lightning_model(torch.FloatTensor(wav).unsqueeze(0).to(device))
emb_wav_cpu = emb_wav.data.cpu().numpy()
np.save(hparams.save_path, emb_wav_cpu[0])

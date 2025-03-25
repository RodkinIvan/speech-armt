import os
import torch
import torchaudio
from tqdm import tqdm
from speechtokenizer import SpeechTokenizer
import moviepy.editor as mp
from pytubefix import Playlist
import torch.nn as nn
import torch
from torch.nn import  functional as F
from mamba_ssm import Mamba
import json
with open("./config.json", "r") as f:
    config = json.load(f)

device = 'cuda'
n_embed = config["n_embed"]
n_heads = config["n_heads"]
n_layers = config["n_layers"]
dropout = config["dropout"]
vocab_size = config["vocab_size"]
sample_rate = config["sample_rate"]
length_of_clips = config["length_of_clips"]
block_size = config["block_size"]
NUM_QUANTIZERS_USED = 4


##########################################
# Speech Tokenizer and Audio Processing
##########################################
def load_speech_tokenizer(config_path, ckpt_path, device):
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
    speech_tokenizer.eval()
    return speech_tokenizer

def normalize_waveform(waveform, sr, target_sr):
    if len(waveform.shape) == 2 and waveform.shape[1] > 0:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.reshape(1, -1)
    waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform

def flatten_tokens(tokens, num_quantizers=NUM_QUANTIZERS_USED):
    n_q, B, T = tokens.shape
    # Transpose tokens so that timesteps are contiguous: a1, b1, c1, a2, b2, c2, ...
    transpose_tokens = tokens.transpose(0, 2)
    return transpose_tokens.reshape(B, T * num_quantizers)

def unflatten_tokens(tokens, num_quantizers=NUM_QUANTIZERS_USED):
    B, L = tokens.shape
    T = L // num_quantizers
    return tokens.reshape(T, B, num_quantizers).transpose(0, 2)

def filter_audio_length(batch, sample_rate, clip_length):
    target_length = sample_rate * clip_length
    return [len(audio[0]) == target_length for audio in batch["audio_array"]]

def tokenize_waveform(speech_tokenizer, waveform):
    # waveforms come in on CPU by default
    waveform = waveform.to("cuda")
    with torch.no_grad():
        codes = speech_tokenizer.encode(waveform.unsqueeze(0))
        semantic_tokens = codes[:NUM_QUANTIZERS_USED, :, :].cpu()
        return flatten_tokens(semantic_tokens, NUM_QUANTIZERS_USED)

def save_waveform(filename, waveform, sample_rate=16000):
    torchaudio.save(filename, waveform[0].detach().cpu(), sample_rate)

def decode_tokens(speech_tokenizer, tokens):
    unflattened = unflatten_tokens(tokens)
    return speech_tokenizer.decode(unflattened)

def save_to_file(speech_tokenizer, tok, filename):
    device = next(speech_tokenizer.parameters()).device
    
    tok = tok.detach().to(device)
    unflattened = unflatten_tokens(tok)
    
    outputwav = speech_tokenizer.decode(unflattened)
    outputwav = outputwav.cpu()
    
    # Save the waveform
    save_waveform(filename, outputwav)

##########################################
# Audio Download and Conversion
##########################################
def download_audio_from_playlist(playlist_url, output_path):
    playlist = Playlist(playlist_url)
    for video in playlist.videos:
        try:
            audio_stream = video.streams.get_audio_only()
            audio_stream.download(output_path=output_path, filename=video.title + ".mp4")
        except Exception as e:
            pass

def convert_mp4_to_wav_clips(mp4_file, output_dir, clip_length=length_of_clips):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio = mp.AudioFileClip(mp4_file)
    duration = int(audio.duration)
    base_name = os.path.basename(mp4_file).split('.')[0].replace(' ', '_').replace("#", "num")
    for start in range(0, duration, clip_length):
        outputpath = os.path.join(output_dir, f'{base_name}_clip_{start}_{start+clip_length}.wav')
        if os.path.exists(outputpath):
            continue
        end = min(start + clip_length, duration)
        clip = audio.subclip(start, end)
        clip.write_audiofile(outputpath, logger=None)
    print(f"Converted {mp4_file} successfully.")

def get_audio_files(directory, extension='.mp4'):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]

##########################################
# Dataset and Data Collator
##########################################
class AudioTokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens_list):
        self.tokens_list = tokens_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return {"tokens": self.tokens_list[idx]}

def data_collator(batch):
    tokens = [item["tokens"] for item in batch]
    batch_max_len = max([t.size(0) for t in tokens])
    padded_tokens = []
    for t in tokens:
        pad_length = batch_max_len - t.size(0)
        if pad_length > 0:
            t = torch.cat([t, torch.zeros(pad_length, dtype=t.dtype)], dim=0)
        padded_tokens.append(t)
    padded_tokens = torch.stack(padded_tokens)  # (B, T)
    input_ids = padded_tokens[:, :-1].contiguous()
    labels = padded_tokens[:, 1:].contiguous()
    return {"input_ids": input_ids, "labels": labels}

##########################################
# Audio Generation Utility
##########################################
def produce_wav(speech_tokenizer, filename, model, example, block_size, device, wandb_obj=None):
    """
    Generate audio autoregressively from a token sequence.
    
    Args:
      speech_tokenizer: The SpeechTokenizer instance.
      filename: Base filename for saving output audio.
      model: The generative model (wrapped for Trainer compatibility).
      example: Token tensor of shape (T,). Generation starts from the first half.
      block_size: Maximum context length used for generation.
      device: Torch device.
      wandb_obj: (Optional) WandB object for logging files.
    
    The function saves two files:
      - <filename>_test.wav : The generated audio.
      - <filename>_input.wav: The original input audio.
    """
    first_half = example.shape[-1] // 2
    tokens = example[:first_half].reshape(1, first_half)
    max_new_tokens = example.shape[-1] - first_half
    idx = tokens.to(device)
    for _ in tqdm(range(max_new_tokens), desc="Generating audio"):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        last_logits = logits["logits"][:, -1, :]#logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=1)
        next_index = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_index), dim=1)
    # Save generated audio and original input for comparison
    save_to_file(speech_tokenizer, idx, f"{filename}_test.wav")
    save_to_file(speech_tokenizer, example.unsqueeze(0), f"{filename}_input.wav")
    if wandb_obj is not None:
        wandb_obj.save(f"{filename}_test.wav")
        wandb_obj.save(f"{filename}_input.wav")


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa_head = Mamba(
            d_model=n_embed,
            d_state=16,
            d_conv=8,
            expand=1,
        ).to(device)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MambaAudioModel(nn.Module):
    def __init__(self, vocab_size = vocab_size, n_embed = n_embed, n_heads = n_heads, n_layers = n_layers, dropout=dropout,
     block_size = block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits_reshaped, targets)
            logits = logits.view(B, T, C)
        return logits, loss

# Wrap model for Trainer compatibility
class HFModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  
    def forward(self, input_ids, labels=None):
        logits, loss = self.model(input_ids, targets=labels)
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output
import os
import json
import torch
import torchaudio
from tqdm import tqdm
import moviepy.editor as mp
from pytubefix import Playlist
from datasets import load_dataset

# Global configuration (you can also choose to pass config as argument)
with open("./config.json", "r") as f:
    config = json.load(f)
device = 'cuda'
NUM_QUANTIZERS_USED = 4

##########################################
# Speech Tokenizer and Audio Processing
##########################################
def load_speech_tokenizer(config_path, ckpt_path, device):
    from speechtokenizer import SpeechTokenizer  # local import if needed
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
    speech_tokenizer.eval()
    return speech_tokenizer

def normalize_waveform(waveform, sr, target_sr):
    if len(waveform.shape) == 2 and waveform.shape[1] > 0:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.reshape(1, -1)
    return torchaudio.functional.resample(waveform, sr, target_sr)

def flatten_tokens(tokens, num_quantizers=NUM_QUANTIZERS_USED):
    n_q, B, T = tokens.shape
    # Transpose so that timesteps are contiguous
    return tokens.transpose(0, 2).reshape(B, T * num_quantizers)

def unflatten_tokens(tokens, num_quantizers=NUM_QUANTIZERS_USED):
    B, L = tokens.shape
    T = L // num_quantizers
    return tokens.reshape(T, B, num_quantizers).transpose(0, 2)

def tokenize_waveform(speech_tokenizer, waveform):
    waveform = waveform.to("cuda")
    with torch.no_grad():
        codes = speech_tokenizer.encode(waveform.unsqueeze(0))
        semantic_tokens = codes[:NUM_QUANTIZERS_USED, :, :].cpu()
    return flatten_tokens(semantic_tokens, NUM_QUANTIZERS_USED)

def save_waveform(filename, waveform, sample_rate=16000):
    torchaudio.save(filename, waveform[0].detach().cpu(), sample_rate)

def decode_tokens(speech_tokenizer, tokens):
    return speech_tokenizer.decode(unflatten_tokens(tokens))

def save_to_file(speech_tokenizer, tok, filename):
    device = next(speech_tokenizer.parameters()).device
    tok = tok.detach().to(device)
    outputwav = decode_tokens(speech_tokenizer, tok).cpu()
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
        except Exception:
            pass

def convert_mp4_to_wav_clips(mp4_file, output_dir, clip_length=config["length_of_clips"]):
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

def process_audio_dataset(data_dir, speech_tokenizer, device, length_of_clips,
                          save_examples=True, num_test_files=10, testfile_dir="testfiles"):
    print("Loading Dataset")
    audio_dataset = load_dataset("audiofolder", data_dir=data_dir)["train"]

    print("Normalizing waveforms")
    audio_dataset = audio_dataset.map(
        lambda x: {
            "original_sampling_rate": x["audio"]["sampling_rate"],
            "audio_array": normalize_waveform(
                torch.tensor(x["audio"]["array"]),
                x["audio"]["sampling_rate"],
                speech_tokenizer.sample_rate,
            ),
        },
        remove_columns=["audio"],
        writer_batch_size=15000,
    )

    print("Filtering dataset for correct clip length")
    audio_dataset = audio_dataset.filter(
        lambda batch: [len(audio[0]) == speech_tokenizer.sample_rate * length_of_clips for audio in batch["audio_array"]],
        batched=True, batch_size=32, num_proc=1
    )

    print("Tokenizing waveforms")
    audio_dataset = audio_dataset.map(
        lambda x: {"tokens": tokenize_waveform(speech_tokenizer, torch.tensor(x["audio_array"]))},
        remove_columns=["audio_array"],
        writer_batch_size=15000,
    )

    if save_examples:
        os.makedirs(testfile_dir, exist_ok=True)
        for idx, t in enumerate(audio_dataset.select(range(0, num_test_files))):
            save_to_file(speech_tokenizer, torch.tensor(t["tokens"]).to(device), f"{testfile_dir}/{idx}_test.wav")

    audio_dataset = audio_dataset.with_format("torch").train_test_split(0.05)
    train_tokens = [ex["tokens"].squeeze(0) for ex in audio_dataset["train"]]
    eval_tokens = [ex["tokens"].squeeze(0) for ex in audio_dataset["test"]]
    from torch.utils.data import Dataset
    class AudioTokenDataset(Dataset):
        def __init__(self, tokens_list):
            self.tokens_list = tokens_list
        def __len__(self):
            return len(self.tokens_list)
        def __getitem__(self, idx):
            return {"tokens": self.tokens_list[idx]}
    return AudioTokenDataset(train_tokens), AudioTokenDataset(eval_tokens)

def load_and_preprocess_dataset(config, speech_tokenizer, device, playlist_url=None):
    if playlist_url:
        download_audio_from_playlist(playlist_url, 'audio/')
        audio_files = get_audio_files('audio')
        for af in tqdm(audio_files):
            convert_mp4_to_wav_clips(af, 'MarcBotClips', config["length_of_clips"])
    return process_audio_dataset("./MarcBotClips", speech_tokenizer, device, config["length_of_clips"])

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


def evaluate_and_generate_audio(trainer, eval_dataset, speech_tokenizer, block_size, device, wandb, num_samples=2):
    print("Evaluating on test dataset...")
    eval_results = trainer.evaluate()
    print("Evaluation Metrics:", eval_results)

    os.makedirs("generated_audio", exist_ok=True)
    for i in range(num_samples):
        example_tokens = eval_dataset[i]["tokens"]
        gen_filename = f"generated_audio/eval_sample_{i}"
        print(f"Generating audio for sample {i}...")
        produce_wav(
            speech_tokenizer, 
            gen_filename, 
            trainer.model, 
            example_tokens, 
            block_size, 
            device,
            wandb_obj=wandb
        )
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import wandb
from tqdm import tqdm




# Import utility functions and classes
from utils import (
    load_speech_tokenizer,
    normalize_waveform,
    tokenize_waveform,
    get_audio_files,
    convert_mp4_to_wav_clips,
    download_audio_from_playlist,
    AudioTokenDataset,
    data_collator,
    save_to_file,
    produce_wav,
    NUM_QUANTIZERS_USED,
    length_of_clips,
    MambaAudioModel,
    HFModelWrapper,
    filter_audio_length
)

#############################
# Hyperparameters & Device  #
#############################
device = torch.device("cuda")
batch_size = 4
epochs = 15
lr = 1e-4
block_size = 2000
n_embed = 512
n_heads = 6
n_layers = 6
dropout = 0.2
vocab_size = 1024

##########################################
# Load SpeechTokenizer and Process Audio
##########################################
# Update these paths as needed
config_path = './SpeechTokenizer/speechtokenizer_hubert_avg/config.json'
ckpt_path = './SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt'
speech_tokenizer = load_speech_tokenizer(config_path, ckpt_path, device)

speech_tokenizer.eval()
# Download and convert audio (if necessary)
playlist_url = "https://www.youtube.com/watch?v=Lp7E973zozc&list=PLQltO7RlbjPJnbfHLsFJWP-DYnWPugUZ7"
download_audio_from_playlist(playlist_url, 'audio/')
audio_files = get_audio_files('audio', extension='.mp4')
for af in tqdm(audio_files):
    convert_mp4_to_wav_clips(af, 'MarcBotClips', LENGTH_OF_CLIPS)

##########################################
# Create and Preprocess Dataset
##########################################
print("Loading Dataset")
audio_dataset = load_dataset("audiofolder", data_dir="./MarcBotClips")["train"]

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
target_sample_rate = speech_tokenizer.sample_rate
target_clip_length = length_of_clips

audio_dataset = audio_dataset.filter(
    lambda batch: filter_audio_length(batch, target_sample_rate, target_clip_length),
    batched=True,
    batch_size=32, 
    num_proc=1      
)

print("Tokenizing waveforms")
audio_dataset = audio_dataset.map(
    lambda x: {"tokens": tokenize_waveform(speech_tokenizer, torch.tensor(x["audio_array"]))},
    remove_columns=["audio_array"],
    writer_batch_size=15000,
)

os.makedirs("testfiles", exist_ok=True)
for idx, t in enumerate(audio_dataset.select(range(0, 10))):
    save_to_file(speech_tokenizer, torch.tensor(t["tokens"]).to(device), f"testfiles/{idx}_test.wav")

audio_dataset = audio_dataset.with_format('torch')
audio_dataset = audio_dataset.train_test_split(0.05)

# Prepare token lists for dataset wrapping
train_tokens = [ex["tokens"].squeeze(0) for ex in audio_dataset["train"]]
eval_tokens  = [ex["tokens"].squeeze(0) for ex in audio_dataset["test"]]
train_dataset = AudioTokenDataset(tokens_list=train_tokens)
eval_dataset  = AudioTokenDataset(tokens_list=eval_tokens)

##########################################
# Define Model Components
##########################################


model_original = MambaAudioModel()
model = HFModelWrapper(model_original)
model.to(device)

##########################################
# Setup Hugging Face Trainer
##########################################
wandb.init(project="speech_test")

training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    learning_rate=lr,
    max_steps=3000,
    save_steps=500,
    save_total_limit=5,
    report_to=["wandb"],
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

##########################################
# Train and Evaluate the Model
##########################################
print("Starting training...")
trainer.train()

print("Evaluating on test dataset...")
eval_results = trainer.evaluate()
print("Evaluation Metrics:", eval_results)

##########################################
# Generate Audio Samples from Evaluation
##########################################

num_samples_to_generate = 2  # for example, generate 2 audio outputs
os.makedirs("generated_audio", exist_ok=True)

for i in range(num_samples_to_generate):
    example_tokens = eval_dataset[i]["tokens"]
    gen_filename = f"generated_audio/eval_sample_{i}"
    print(f"Generating audio for sample {i}...")
    produce_wav(
        speech_tokenizer, 
        gen_filename, 
        model, 
        example_tokens, 
        block_size, 
        device,
        wandb_obj=wandb  


    )
wandb.finish()
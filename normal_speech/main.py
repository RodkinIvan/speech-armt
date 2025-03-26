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
with open("./config.json", "r") as f:
    config = json.load(f)

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
# Load SpeechTokenizer and Process Audio
##########################################
# Update these paths as needed
config_path = './SpeechTokenizer/speechtokenizer_hubert_avg/config.json'
ckpt_path = './SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt'
speech_tokenizer = load_speech_tokenizer(config_path, ckpt_path, device)
speech_tokenizer.eval()

playlist_url = "https://www.youtube.com/watch?v=Lp7E973zozc&list=PLQltO7RlbjPJnbfHLsFJWP-DYnWPugUZ7"
download_audio_from_playlist(playlist_url, 'audio/')
audio_files = get_audio_files('audio', extension='.mp4')
for af in tqdm(audio_files):
    convert_mp4_to_wav_clips(af, 'MarcBotClips', LENGTH_OF_CLIPS)

##########################################
# Create and Preprocess Dataset
##########################################
train_dataset, eval_dataset = process_audio_dataset(
    data_dir="./MarcBotClips",
    speech_tokenizer=speech_tokenizer,
    length_of_clips=length_of_clips,
    device=device,
)
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
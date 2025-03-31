import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoConfig

import wandb
from tqdm import tqdm
import json

from utils import (
    load_speech_tokenizer,
    normalize_waveform,
    tokenize_waveform,
    get_audio_files,
    process_audio_dataset,
    convert_mp4_to_wav_clips,
    download_audio_from_playlist,
    load_and_preprocess_dataset,
    AudioTokenDataset,
    data_collator,
    save_to_file,
    produce_wav,
    NUM_QUANTIZERS_USED,
    length_of_clips,
    MambaAudioModel,
    HFModelWrapper,
    filter_audio_length,
    initialize_model,
    evaluate_and_generate_audio
)

#############################
# Hyperparameters & Device  #
#############################
device = torch.device("cuda")
with open("./config.json", "r") as f:
        config = json.load(f)

length_of_clips = config["length_of_clips"]
#Load dataset
config_path = './SpeechTokenizer/speechtokenizer_hubert_avg/config.json'
ckpt_path = './SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt'
speech_tokenizer = load_speech_tokenizer(config_path, ckpt_path, device)
speech_tokenizer.eval()

url = 'https://www.youtube.com/watch?v=a7fzkqLozwA&list=PLnuc7k2Czju3daT5xsUrli-cmGxQEU08H'#"https://www.youtube.com/watch?v=Lp7E973zozc&list=PLQltO7RlbjPJnbfHLsFJWP-DYnWPugUZ7"
train_dataset, eval_dataset = load_and_preprocess_dataset(config, speech_tokenizer, device)#, playlist_url= url)

model_name = 'mamba'  # [mamba, gptneox, armt]
model = initialize_model(model_name, config)

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


evaluate_and_generate_audio(trainer, eval_dataset, speech_tokenizer, config["block_size"], device)
wandb.finish()
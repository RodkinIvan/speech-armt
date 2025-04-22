import os
import sys
import argparse
import json
import torch
import wandb
from transformers import TrainingArguments, AutoConfig
from tqdm import tqdm

from data_utils import load_speech_tokenizer, load_and_preprocess_dataset, load_sample_data, data_collator, evaluate_and_generate_audio
from model_utils import initialize_model, ForceEvalLossCallback
from trainer_utils import CustomTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech model")
    parser.add_argument("--config_path", type=str, default="./config.json", help="Path to config JSON file")
    parser.add_argument("--tokenizer_config", type=str, default="./SpeechTokenizer/speechtokenizer_hubert_avg/config.json", help="Path to speech tokenizer config")
    parser.add_argument("--tokenizer_ckpt", type=str, default="./SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt", help="Path to speech tokenizer checkpoint")
    parser.add_argument("--model_name", type=str, default="armt", choices=["mamba", "gptneox", "armt"], help="Model branch to use")
    parser.add_argument("--playlist_url", type=str, default=None, help="Optional YouTube playlist URL for audio download")
    parser.add_argument("--use_sample_data", action="store_true", help="If set, load a partial HF dataset instead of local files")
    parser.add_argument("--dataset_name", type=str, default="parler-tts/mls_eng", help="HF dataset name to load sample data from")
    parser.add_argument("--train_samples", type=int, default=500, help="Number of training samples to load")
    parser.add_argument("--dev_samples", type=int, default=50, help="Number of dev samples to load")
    parser.add_argument("--test_samples", type=int, default=5, help="Number of test samples to load")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed setting")
    return parser.parse_args()

def main():
    def seed_everything(seed: int):
        import random, os
        import numpy as np
        import torch

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    args = parse_args()
    seed_everything(args.random_seed)
    
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length_of_clips = config["length_of_clips"]

    speech_tokenizer = load_speech_tokenizer(args.tokenizer_config, args.tokenizer_ckpt, device)
    speech_tokenizer.eval()

    if args.use_sample_data:
        train_data, dev_data, test_data = load_sample_data(
            dataset_name=args.dataset_name,
            train_samples=args.train_samples,
            dev_samples=args.dev_samples,
            test_samples=args.test_samples,
            speech_tokenizer=speech_tokenizer
        )
        eval_data = dev_data + test_data
        from data_utils import AudioTokenDataset
        train_dataset = AudioTokenDataset([d["tokens"] for d in train_data])
        eval_dataset = AudioTokenDataset([d["tokens"] for d in eval_data])
    else:
        train_dataset, eval_dataset = load_and_preprocess_dataset(config, speech_tokenizer, device, playlist_url=args.playlist_url)

    # Initialize model
    model = initialize_model(args.model_name, config)

    wandb.init(project="test_3")
    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=100,
        learning_rate=config['learning_rate'],
        max_steps=1000,
        save_steps=50,
        save_total_limit=5,
        report_to=["wandb"],
        remove_unused_columns=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(ForceEvalLossCallback(trainer, eval_dataset))

    print("Starting training...")
    trainer.train()

    evaluate_and_generate_audio(trainer, eval_dataset, speech_tokenizer, config["block_size"], device, wandb=wandb.run)
    wandb.finish()

if __name__ == "__main__":
    main()

#test case for load dataset from HF: python main.py --use_sample_data --dataset_name "parler-tts/mls_eng" --train_samples 5 --dev_samples 2 --test_samples 1 --random_seed 42
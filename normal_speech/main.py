import os
import sys
import argparse
import json
import torch
from transformers import TrainingArguments, AutoConfig
import wandb
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from data_utils import load_speech_tokenizer, load_and_preprocess_dataset, data_collator, evaluate_and_generate_audio
from model_utils import initialize_model
from trainer_utils import CustomTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a speech model")
    parser.add_argument("--config_path", type=str, default="./config.json", help="Path to config JSON file")
    parser.add_argument("--tokenizer_config", type=str, default="./SpeechTokenizer/speechtokenizer_hubert_avg/config.json", help="Path to speech tokenizer config")
    parser.add_argument("--tokenizer_ckpt", type=str, default="./SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt", help="Path to speech tokenizer checkpoint")
    parser.add_argument("--model_name", type=str, default="armt", choices=["mamba", "gptneox", "armt"], help="Model branch to use")
    parser.add_argument("--playlist_url", type=str, default=None, help="Optional YouTube playlist URL to download audio")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training and evaluation (overrides config)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    if args.num_train_epochs is not None:
        config["epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["per_device_train_batch_size"] = args.batch_size
        config["per_device_eval_batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length_of_clips = config["length_of_clips"]

    speech_tokenizer = load_speech_tokenizer(args.tokenizer_config, args.tokenizer_ckpt, device)
    speech_tokenizer.eval()

    train_dataset, eval_dataset = load_and_preprocess_dataset(config, speech_tokenizer, device, playlist_url=args.playlist_url)

    model = initialize_model(args.model_name, config)

    wandb.init(project="speech_test")

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        evaluation_strategy="steps",
        eval_steps=300,
        logging_steps=100,
        learning_rate=config['learning_rate'],
        max_steps=400,
        save_steps=200,
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

    print("Starting training...")
    trainer.train()

    # Evaluate and generate audio examples
    evaluate_and_generate_audio(trainer, eval_dataset, speech_tokenizer, config["block_size"], device, wandb=wandb.run)
    wandb.finish()

if __name__ == "__main__":
    main()

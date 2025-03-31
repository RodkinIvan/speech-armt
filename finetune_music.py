import os
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, GPTNeoXTokenizerFast, AutoTokenizer
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
import librosa
import wandb
import argparse
from pathlib import Path
import logging
import pickle
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Train GPTNeoX on music data with pretrained tokenizer')
parser.add_argument('--dataset_path', type=str, default='irodkin/song_structure_with_test', 
                    help='Path to the dataset')
parser.add_argument('--model_cfg', type=str, 
                    help='Base model config')
parser.add_argument('--tokenizer_name', type=str, default='facebook/encodec_24khz', 
                    help='Pretrained tokenizer to use (encodec, hubert, or other audio model)')
parser.add_argument('--output_dir', type=str, default='./music_model_output', 
                    help='Output directory for model checkpoints')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--iters', type=int, default=1000, help='Number of batches in training.')
parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
parser.add_argument('--tokenizer_type', type=str, default='encodec', 
                    choices=['encodec', 'hubert', 'wav2vec', 'vq_vae', 'wavtokenizer'],
                    help='Type of pretrained tokenizer to use')
parser.add_argument('--sample_rate', type=int, default=24000, help='Audio sample rate')
parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
parser.add_argument('--cache_dir', type=str, default='./tokenizer_cache', 
                    help='Directory to cache tokenized features')
parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate tokenizer performance')
args = parser.parse_args()

def load_music_dataset(dataset_path):
    """Load music dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Check if it's a Hugging Face dataset ID or a local path
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    
    logger.info(f"Dataset loaded with {len(dataset['train'])} training examples")
    return dataset

class PretrainedAudioTokenizer:
    def __init__(self, tokenizer_type, tokenizer_name, args):
        self.tokenizer_type = tokenizer_type
        self.tokenizer_name = tokenizer_name
        self.args = args
        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing {tokenizer_type} tokenizer: {tokenizer_name}")
        
        if tokenizer_type == 'encodec':
            # For EnCodec, we'll use the pretrained model from torchaudio or transformers
            try:
                import torchaudio
                from torchaudio.models import encodec
                self.model = encodec.EncodecModel.encodec_model_24khz()
                self.model.eval()
                logger.info(f"Loaded EnCodec model from torchaudio")
            except ImportError:
                # Fall back to HuggingFace transformers implementation
                from transformers import EncodecModel
                self.model = EncodecModel.from_pretrained(tokenizer_name)
                self.model.eval()
                logger.info(f"Loaded EnCodec model from transformers")
                
        elif tokenizer_type == 'hubert':
            # For HuBERT, we'll use the pretrained model for feature extraction
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tokenizer_name)
            self.model = HubertModel.from_pretrained(tokenizer_name)
            self.model.eval()
            
        elif tokenizer_type == 'wav2vec':
            # For Wav2Vec2, similar to HuBERT
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tokenizer_name)
            self.model = Wav2Vec2Model.from_pretrained(tokenizer_name)
            self.model.eval()
            
        elif tokenizer_type == 'vq_vae':
            # For a VQ-VAE model (like DALL-E or VQGAN for audio)
            # This would typically use a custom implementation or specific audio VQ-VAE
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(tokenizer_name)
            self.model.eval()
            
        elif tokenizer_type == 'wavtokenizer':
            # For novateur/WavTokenizer
            # Parse args.tokenizer_name as "config_path,model_path"
            config_path, model_path = tokenizer_name.split(',')

            # os.chdir('./WavTokenizer')
            from decoder.pretrained import WavTokenizer
            self.model = WavTokenizer.from_pretrained0802(config_path, model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Import the audio conversion utility
            from encoder.utils import convert_audio
            # os.chdir('..')

            self.convert_audio = convert_audio
            
            logger.info(f"Loaded WavTokenizer model from {model_path}")
            # Default bandwidth_id (can be made configurable)
            self.bandwidth_id = torch.tensor([0]).to(self.device)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        # For GPT training, we need a text tokenizer for the output format
        
        # Get the vocabulary size from the model
        if tokenizer_type == 'wavtokenizer':
            # WavTokenizer typically has a different vocabulary size concept
            # This would need to be determined from the model configuration
            # For now, using a default value based on typical codebook sizes
            self.vocab_size = 4096  # Can be adjusted based on specific WavTokenizer configuration
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
            self.vocab_size = self.model.config.vocab_size
        else:
            # Default for most audio tokenizers
            self.vocab_size = 1024

    def tokenize_audio(self, audio_data):
        """Tokenize audio data using the pretrained model"""
        # Check cache first
        cache_key = str(hash(str(audio_data)))
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Convert to appropriate format (typically float32 tensor)

        name = audio_data['path']
        sampling_rate = audio_data['sampling_rate']
        audio_data = audio_data['array']

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        
        # Resample if needed
        current_sample_rate = self.args.sample_rate
        target_sample_rate = self.args.sample_rate  # Can be adjusted based on model requirements
        
        if current_sample_rate != target_sample_rate and self.tokenizer_type != 'wavtokenizer':
            import torchaudio.functional as F
            audio_tensor = F.resample(audio_tensor, 
                                     orig_freq=current_sample_rate, 
                                     new_freq=target_sample_rate)
        
        with torch.no_grad():
            if self.tokenizer_type == 'encodec':
                # EnCodec specific processing
                # The model returns the encoded indices directly
                try:
                    encoded_frames = self.model.encode(audio_tensor)
                    codes = encoded_frames[0][0]  # Get the quantized codes
                    # Flatten the codes (removing batch dim)
                    tokens = codes.flatten().tolist()
                except Exception as e:
                    logger.error(f"Error in EnCodec tokenization: {e}")
                    # Fallback to using a simple spectrogram approach
                    spec = librosa.feature.melspectrogram(y=np.array(audio_data), 
                                                        sr=self.args.sample_rate,
                                                        n_mels=80)
                    log_spec = librosa.power_to_db(spec)
                    # Quantize to get discrete tokens
                    tokens = np.floor((log_spec - log_spec.min()) / 
                                     (log_spec.max() - log_spec.min()) * 
                                     (self.vocab_size - 1)).astype(int)
                    tokens = tokens.flatten().tolist()
                
            elif self.tokenizer_type in ['hubert', 'wav2vec']:
                # HuBERT/Wav2Vec2 processing
                # First preprocess with the feature extractor
                inputs = self.feature_extractor(audio_tensor.squeeze().numpy(), 
                                              return_tensors="pt", 
                                              sampling_rate=target_sample_rate)
                
                # Get the model output
                outputs = self.model(**inputs)
                
                # Get the final hidden states
                hidden_states = outputs.last_hidden_state
                
                # Quantize the continuous representations to get discrete tokens
                # This is a simple approach using k-means or similar clustering
                # For demonstration, we'll just do a simple discretization
                hidden_norm = (hidden_states - hidden_states.min()) / (hidden_states.max() - hidden_states.min())
                tokens = (hidden_norm * (self.vocab_size - 1)).round().int()
                tokens = tokens.flatten().tolist()
                
            elif self.tokenizer_type == 'vq_vae':
                # VQ-VAE processing
                # Typically returns discrete indices from the codebook
                outputs = self.model(audio_tensor)
                
                # Depending on the model, get the quantized indices
                if hasattr(outputs, 'quantized_indices'):
                    tokens = outputs.quantized_indices.flatten().tolist()
                else:
                    # Fallback similar to above
                    hidden_states = outputs.last_hidden_state
                    tokens = ((hidden_states - hidden_states.min()) / 
                            (hidden_states.max() - hidden_states.min()) * 
                            (self.vocab_size - 1)).round().int()
                    tokens = tokens.flatten().tolist()
                    
            elif self.tokenizer_type == 'wavtokenizer':
                # WavTokenizer specific processing
                try:
                    # Use WavTokenizer's specific audio conversion
                    wav = audio_tensor
                    if wav.size(0) > 1:  # If more than one channel (stereo)
                        wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
                    
                    # Convert audio using WavTokenizer's utility
                    wav = self.convert_audio(wav, current_sample_rate, 24000, 1)
                    wav = wav.to(self.device)
                    
                    # Encode using WavTokenizer
                    features, discrete_code = self.model.encode_infer(wav, bandwidth_id=self.bandwidth_id)
                    
                    # Use discrete_code as tokens
                    # The exact format depends on WavTokenizer's implementation
                    # Typically, discrete_code is already the tokens we need
                    if isinstance(discrete_code, list):
                        # Handle multiple codebook case
                        all_tokens = []
                        for code in discrete_code:
                            all_tokens.extend(code.flatten().cpu().tolist())
                        tokens = all_tokens
                    else:
                        # Single codebook case
                        tokens = discrete_code
                except Exception as e:
                    logger.error(f"Error in WavTokenizer processing: {e}")
                    # Fallback to a simple approach
                    mel_spec = librosa.feature.melspectrogram(
                        y=np.array(audio_data),
                        sr=current_sample_rate,
                        n_mels=80
                    )
                    log_mel = librosa.power_to_db(mel_spec)
                    tokens = np.floor((log_mel - log_mel.min()) / 
                                     (log_mel.max() - log_mel.min()) * 
                                     (self.vocab_size - 1)).astype(int)
                    tokens = tokens.flatten().tolist()
            
            # Truncate if too long
            if len(tokens) > self.args.max_length:
                tokens = tokens[:self.args.max_length]
            
            # Cache the result
            with open(cache_file, 'wb') as f:
                pickle.dump(tokens, f)
            
            return tokens

    def decode_audio(self, tokens):
        """Attempt to decode tokens back to audio (for evaluation)"""
        if self.tokenizer_type == 'encodec':
            # Reshape tokens to expected format
            try:
                # Assume the shape expected by the decoder
                # This depends on the specific EnCodec model parameters
                batch_size = 1
                sequence_length = len(tokens) // 8  # Typical EnCodec has 8 codebooks
                codebook_size = 1024
                
                # Reshape tokens
                tokens_tensor = torch.tensor(tokens, dtype=torch.long)
                tokens_tensor = tokens_tensor.reshape(batch_size, 8, sequence_length)
                
                # Decode
                with torch.no_grad():
                    audio_out = self.model.decode([(tokens_tensor, None)])
                
                return audio_out.squeeze().numpy()
            except Exception as e:
                logger.error(f"Error in EnCodec decoding: {e}")
                return None
                
        elif self.tokenizer_type == 'wavtokenizer':
            try:
                # Reshape tokens back to the format expected by WavTokenizer
                # This depends on the specific WavTokenizer implementation
                # The exact reshaping will depend on how the tokens were extracted
                
                # For demonstration, assuming tokens can be reshaped into features
                # directly. In practice, this might require additional processing.
                tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(self.device)
                
                # Attempt to reconstruct features from tokens
                # The exact method depends on WavTokenizer implementation
                # This is a placeholder and would need to be adapted
                if hasattr(self.model, 'decode_from_tokens'):
                    # If there's a direct decoding method from tokens
                    audio_out = self.model.decode_from_tokens(tokens_tensor, self.bandwidth_id)
                else:
                    # Otherwise try to reconstruct features first
                    # This is a simplification and may not work directly
                    features = self.model.codes_to_features(tokens_tensor)

                    audio_out = self.model.decode(features, bandwidth_id=self.bandwidth_id)
                
                return audio_out.squeeze().cpu().numpy()
            except Exception as e:
                logger.error(f"Error in WavTokenizer decoding: {e}")
                return None
        else:
            # For most models, direct decoding to audio is not easily available
            # Would require additional training or processing
            logger.warning(f"Direct audio decoding not implemented for {self.tokenizer_type}")
            return None

def prepare_dataset_for_training(dataset, tokenizer, args):
    """Prepare dataset for model training"""
    logger.info("Preparing dataset for training...")
    
    def tokenize_function(examples):
        tokenized_inputs = []
        attention_masks = []
        
        # attention_masks = torch.ones_like(codes)
        for audio in examples["audio"]:
            # tokens = tokenizer.tokenize_audio(audio)
            # tokenized_inputs.append(tokens)
            # attention_masks.append([1] * len(tokens))
            wav, sr = audio['array'], audio['sampling_rate']
            wav = torch.tensor(wav).unsqueeze(0).to(torch.float32)
            wav = tokenizer.convert_audio(wav, sr, 24000, 1).to(tokenizer.device)
            _, codes = tokenizer.model.encode_infer(wav, bandwidth_id=tokenizer.bandwidth_id)
            tokenized_inputs.append(codes)
            attention_masks.append([1] * len(codes))
        
        # Prepare the output format for the model
        result = {
            "input_ids": tokenized_inputs,
            "attention_mask": attention_masks,
        }
        return result
    
    # Tokenize the datasets
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=16,
    )
    
    return tokenized_dataset

def collate_fn(batch):
    
    min_l = min([len(b['input_ids'][0][0]) for b in batch])
    logger.info(min_l)
    assert all([torch.tensor(b['input_ids']).shape[:2] == (1, 1) for b in batch])
    input_ids = [torch.tensor(b['input_ids'])[0, 0, :min_l] for b in batch]
    input_ids = torch.stack(input_ids, dim=0)
    labels = input_ids

    return dict(
        input_ids=input_ids,
        labels=labels
    )
def evaluate_tokenizer_quality(dataset, tokenizer, args):
    """Evaluate the quality of tokenization by reconstruction when possible"""
    logger.info(f"Evaluating {args.tokenizer_type} tokenizer quality...")
    
    # Sample a few audio files
    sample_size = min(5, len(dataset['train']))
    metrics = []
    
    for i in range(sample_size):
        original_audio = dataset['train'][i]['audio']
        
        # Tokenize
        tokens = tokenizer.tokenize_audio(original_audio)
        
        # For evaluation purposes - attempt reconstruction if possible
        reconstructed_audio = tokenizer.decode_audio(tokens)
        
        metric_item = {
            'sample_id': i,
            'num_tokens': len(tokens),
            'compression_ratio': len(tokens) / len(original_audio),
            'tokenizer_type': args.tokenizer_type,
            'vocab_utilization': len(set(tokens)) / tokenizer.vocab_size,
        }
        
        if reconstructed_audio is not None:
            # Calculate reconstruction error
            # Make sure lengths match
            min_len = min(len(original_audio), len(reconstructed_audio))
            original_trim = original_audio['array'][:min_len]
            reconstructed_trim = reconstructed_audio[:min_len]
            
            # Calculate metrics
            mse = np.mean((original_trim - reconstructed_trim)**2)
            # Signal-to-noise ratio
            eps = 1e-10  # Avoid division by zero
            snr = 10 * np.log10(np.sum(original_trim**2) / (np.sum((original_trim - reconstructed_trim)**2) + eps))
            
            metric_item.update({
                'reconstruction_mse': float(mse),
                'reconstruction_snr': float(snr)
            })
            
            # Save original and reconstructed audio for manual inspection
            os.makedirs("./audio_eval", exist_ok=True)
            sf.write(f"./audio_eval/original_{i}.wav", original_trim, args.sample_rate)
            sf.write(f"./audio_eval/reconstructed_{i}.wav", reconstructed_trim, args.sample_rate)
        
        metrics.append(metric_item)
    
    # Log results
    df = pd.DataFrame(metrics)
    logger.info(f"Tokenizer evaluation results:\n{df.describe()}")
    
    # Save metrics to CSV
    df.to_csv(f"./tokenizer_eval_{args.tokenizer_type}.csv", index=False)
    
    return df

def train_model(tokenized_dataset, tokenizer, args):
    """Train the GPTNeoX model"""

    # train_dataloader = DataLoader(tokenized_dataset['train'], collate_fn=collate_fn, batch_size=1)
    logger.info("Initializing model for training...")
    
    # Initialize wandb for tracking
    wandb.init(project="music-generation", name=f"gptneox-{args.tokenizer_type}")
    
    # Define model configuration
    config = GPTNeoXConfig.from_pretrained(args.model_cfg)
    # Set vocabulary size based on tokenizer
    config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    model = GPTNeoXForCausalLM(config=config)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps", 
        eval_steps=500,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=True,
        report_to="wandb",
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=collate_fn
    )
    
    # Train model
    logger.info("Starting model training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(f"{args.output_dir}/{args.tokenizer_type}")
    logger.info(f"Model saved to {args.output_dir}/{args.tokenizer_type}")
    
    # Close wandb run
    wandb.finish()

def generate_music(model, tokenizer, prompt=None, max_length=1024):
    """Generate new music using the trained model"""
    model.eval()
    
    # Start with a prompt, or use a random seed
    if prompt is not None:
        input_ids = tokenizer.tokenize_audio(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
    else:
        # Start with a special token or random seed
        input_tensor = torch.tensor([[0]], dtype=torch.long)  # Use an appropriate start token
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
        )
    
    # Decode back to audio
    generated_tokens = outputs[0].tolist()
    audio = tokenizer.decode_audio(generated_tokens)
    
    return audio, generated_tokens

def main():
    """Main training function"""
    # Load dataset
    dataset = load_music_dataset(args.dataset_path)
    
    # Initialize tokenizer
    tokenizer = PretrainedAudioTokenizer(args.tokenizer_type, args.tokenizer_name, args)
    
    # Evaluate tokenizer quality
    tokenizer_metrics = evaluate_tokenizer_quality(dataset, tokenizer, args)
    
    if args.evaluate_only:
        logger.info("Tokenizer evaluation complete. Exiting.")
        return
    
    # Prepare dataset for training
    tokenized_dataset = prepare_dataset_for_training(dataset, tokenizer, args)
    
    # Train model
    train_model(tokenized_dataset, tokenizer, args)
    
    # Example of generating new music (would run after training)
    # if os.path.exists(f"{args.output_dir}/{args.tokenizer_type}"):
    #     logger.info("Loading trained model for music generation...")
    #     model = GPTNeoXForCausalLM.from_pretrained(f"{args.output_dir}/{args.tokenizer_type}")
        
    #     # Generate a sample
    #     sample_audio = dataset['train'][0]['audio']
    #     sample_audio['array'] = sample_audio['array'][:5000]  # Use first 5000 samples as prompt
    #     generated_audio, _ = generate_music(model, tokenizer, prompt=sample_audio)
        
    #     if generated_audio is not None:
    #         os.makedirs("./generated", exist_ok=True)
    #         sf.write("./generated/sample.wav", generated_audio, args.sample_rate)
    #         logger.info("Generated sample saved to ./generated/sample.wav")
    
    logger.info("Process complete!")

if __name__ == "__main__":
    main()
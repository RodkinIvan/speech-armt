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
from armt.model import AssociativeMemoryCell, AssociativeRecurrentWrapper
from transformers.integrations import WandbCallback
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import json
from types import SimpleNamespace
from mamba.model import MambaAudioModel

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
parser.add_argument('--model_name', type=str, default='gptneox', 
                    help='Model name from [gptneox, armt, mamba]')

parser.add_argument('--num_mem_tokens', type=int, default=16, help='armt parameter')
parser.add_argument('--d_mem', type=int, default=32, help='armt parameter')
parser.add_argument('--segment_size', type=int, default=1024, help='armt parameter')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('input_segment_borders', action='store_true', help='Input segment borders for training')

parser.add_argument('--use_equal_segments', action='store_true', help='Use equal segments for training', default=False)
parser.add_argument('--early_stopping_steps', type=int, default=None, 
                    help='Number of validation steps to wait before early stopping')
args = parser.parse_args()

model_name = args.model_name
segment_size = args.segment_size
sample_rate = args.sample_rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_music_dataset(dataset_path):
    """Load music dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Check if it's a Hugging Face dataset ID or a local path
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    
    # dataset['train'] = dataset['train'].select(range(32))
    # dataset['validation'] = dataset['validation'].select(range(1))
    # dataset['test'] = dataset['test'].select(range(1))

    assert 'validation' in dataset
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
                # Validate tokens
                if not isinstance(tokens, list):
                    logger.error("Tokens must be a list")
                    return None
                    
                if not tokens:
                    logger.error("Empty token list")
                    return None
                    
                # Process tokens in smaller chunks to avoid memory issues
                chunk_size = 1024  # Adjust this based on your GPU memory
                audio_chunks = []
                
                for i in range(0, len(tokens), chunk_size):
                    chunk = tokens[i:i + chunk_size]
                    
                    # Validate chunk
                    if not all(isinstance(t, (int, np.integer)) for t in chunk):
                        logger.error("Tokens must be integers")
                        return None
                        
                    # Convert to tensor and ensure proper shape
                    tokens_tensor = torch.tensor(chunk, dtype=torch.long)
                    tokens_tensor = tokens_tensor.unsqueeze(0)  # Add batch dimension
                    tokens_tensor = tokens_tensor.to(self.device)
                    
                    # Clear GPU cache before processing each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    try:
                        # Attempt to reconstruct features from tokens
                        if hasattr(self.model, 'decode_from_tokens'):
                            # If there's a direct decoding method from tokens
                            audio_chunk = self.model.decode_from_tokens(tokens_tensor, self.bandwidth_id)
                        else:
                            # Otherwise try to reconstruct features first
                            features = self.model.codes_to_features(tokens_tensor)
                            audio_chunk = self.model.decode(features, bandwidth_id=self.bandwidth_id)
                        
                        # Move chunk to CPU and convert to numpy
                        audio_chunk = audio_chunk.squeeze().cpu().numpy()
                        audio_chunks.append(audio_chunk)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {i//chunk_size}: {str(e)}")
                        return None
                
                # Concatenate all chunks
                if audio_chunks:
                    return np.concatenate(audio_chunks)
                return None
                
            except Exception as e:
                logger.error(f"Error in WavTokenizer decoding: {e}")
                return None
        else:
            # For most models, direct decoding to audio is not easily available
            # Would require additional training or processing
            logger.warning(f"Direct audio decoding not implemented for {self.tokenizer_type}")
            return None

def prepare_dataset_for_training(dataset, tokenizer, args):
    """Prepare dataset for training"""
    logger.info("Preparing dataset for training...")
    
    def tokenize_function(examples):
        tokenized_inputs = []
        attention_masks = []
        array_lens = []
        onset_times = []
        
        for i, audio in enumerate(examples["audio"]):
            wav, sr = audio['array'], audio['sampling_rate']
            wav = torch.tensor(wav).unsqueeze(0).to(torch.float32)
            wav = tokenizer.convert_audio(wav, sr, 24000, 1).to(tokenizer.device)
            _, codes = tokenizer.model.encode_infer(wav, bandwidth_id=tokenizer.bandwidth_id)
            tokenized_inputs.append(codes)
            attention_masks.append([1] * len(codes))
            array_lens.append(len(audio['array']))
            onset_times.append(examples["label"][i]["onset_time"])
        
        # Prepare the output format for the model
        result = {
            "input_ids": tokenized_inputs,
            "attention_mask": attention_masks,
            "label": [
                {
                    "onset_time": onset_times[i],
                    "array_len": array_lens[i]
                }
                for i in range(len(onset_times))
            ]
        }
        return result
    
    # Tokenize the datasets
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=16,
    )
    return tokenized_dataset

def collate_fn(batch, model_name='gptneox'):
    if model_name != 'armt' or args.use_equal_segments:
        # Original collate_fn for non-ARMT models
        min_l = min([len(b['input_ids'][0][0]) for b in batch])
        input_ids = [torch.tensor(b['input_ids'])[0, 0, :min_l] for b in batch]
        input_ids = torch.stack(input_ids, dim=0)
        labels = input_ids
        return dict(
            input_ids=input_ids,
            labels=labels
        )
    
    # ARMT-specific collate_fn with segmentation
    min_l = min([len(b['input_ids'][0][0]) for b in batch])
    
    # Process each sample in the batch
    all_segments = []
    all_labels = []
    
    for b in batch:
        # Get the input tokens and onset times
        input_tokens = torch.tensor(b['input_ids'])[0, 0, :min_l]
        onset_times = b['label']['onset_time']  # List of onset times in original audio
        
        # Calculate the conversion ratio based on audio array length and token sequence length
        audio_length = b['label']['array_len']  # Length of original audio array
        token_length = len(input_tokens)  # Length of token sequence
        time_to_token_ratio = token_length / audio_length
        
        # Convert onset times to token positions
        token_positions = [int(t * time_to_token_ratio) for t in onset_times]
        token_positions = [p for p in token_positions if p < min_l]  # Filter out positions beyond sequence length
        
        # Add start and end positions
        token_positions = [0] + token_positions + [min_l]
        
        # Split the sequence into segments
        segments = []
        label_segments = []
        for i in range(len(token_positions) - 1):
            start = token_positions[i]
            end = token_positions[i + 1]
            segment = input_tokens[start:end]
            label_segment = input_tokens[start:end]  # Labels are the same as input tokens
            segments.append(segment)
            label_segments.append(label_segment)
        
        all_segments.append(segments)
        all_labels.append(label_segments)
    
    # Find the maximum segment length for each position
    max_segment_lengths = []
    # First find the maximum number of segments across all samples
    max_num_segments = max([len(segments) for segments in all_segments])
    
    # Then for each segment position, find the maximum length
    for i in range(max_num_segments):
        # Only consider samples that have this segment
        segment_lengths = [len(sample_segments[i]) for sample_segments in all_segments if i < len(sample_segments)]
        if segment_lengths:  # Only if we have any samples with this segment
            max_len = max(segment_lengths)
            max_segment_lengths.append(max_len)
    
    # Pad and stack segments, and create attention masks
    padded_segments = []
    padded_labels = []
    attention_masks = []
    labels_masks = []
    for i in range(len(max_segment_lengths)):
        # Pad and stack segments at position i with left padding
        padded_segment = torch.stack([
            torch.nn.functional.pad(
                sample_segments[i],
                (max_segment_lengths[i] - len(sample_segments[i]), 0),  # Left padding
                mode='constant',
                value=0  # Assuming 0 is your padding token
            ) if i < len(sample_segments) else
            torch.zeros(max_segment_lengths[i], dtype=torch.long)  # Create zero tensor for missing segments
            for sample_segments in all_segments
        ])
        padded_segments.append(padded_segment)
        
        # Pad and stack label segments at position i with left padding
        padded_label = torch.stack([
            torch.nn.functional.pad(
                sample_labels[i],
                (max_segment_lengths[i] - len(sample_labels[i]), 0),  # Left padding
                mode='constant',
                value=0  # Assuming 0 is your padding token
            ) if i < len(sample_labels) else
            torch.zeros(max_segment_lengths[i], dtype=torch.long)  # Create zero tensor for missing segments
            for sample_labels in all_labels
        ])
        padded_labels.append(padded_label)
        
        # Create attention mask for this segment position with left padding
        attention_mask = torch.stack([
            torch.cat([
                torch.zeros(max_segment_lengths[i] - len(sample_segments[i])),
                torch.ones(len(sample_segments[i]))
            ]) if i < len(sample_segments) else
            torch.zeros(max_segment_lengths[i])  # All zeros for missing segments
            for sample_segments in all_segments
        ])
        attention_masks.append(attention_mask)

        # Create labels mask for this segment position with left padding
        labels_mask = torch.stack([
            torch.cat([
                torch.zeros(max_segment_lengths[i] - len(sample_labels[i]), dtype=torch.bool),
                torch.ones(len(sample_labels[i]), dtype=torch.bool)
            ]) if i < len(sample_labels) else
            torch.zeros(max_segment_lengths[i], dtype=torch.bool)  # All zeros for missing segments
            for sample_labels in all_labels
        ])
        labels_masks.append(labels_mask)
    
    return {
        'input_ids': padded_segments,  # List of tensors, one for each segment
        'attention_mask': attention_masks,  # List of attention masks, one for each segment
        'labels': padded_labels,  # List of tensors, one for each segment
        'labels_mask': labels_masks,  # List of labels masks, one for each segment
        'input_segmented': True  # Tell the model to expect segmented input
    }

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

def load_model(model_name, tokenizer):
    if model_name == 'gptneox':
        config = GPTNeoXConfig.from_pretrained(args.model_cfg)
        config.vocab_size = tokenizer.vocab_size
        model = GPTNeoXForCausalLM(config=config)
        return model
    elif model_name == 'armt':
        config = GPTNeoXConfig.from_pretrained(args.model_cfg)
        config.vocab_size = tokenizer.vocab_size
        base_model = GPTNeoXForCausalLM(config=config)
        # Initialize ARMT model
        mem_cell_args = dict(
            base_model=base_model,
            num_mem_tokens=args.num_mem_tokens,
            d_mem=args.d_mem, 
            layers_attr="gpt_neox.layers",
        )
        memory_cell = AssociativeMemoryCell(**mem_cell_args)

        model = AssociativeRecurrentWrapper(
            memory_cell=memory_cell,
            segment_size=args.segment_size,
        )
        return model
    elif model_name == 'mamba':
        # Load Mamba config from JSON file
        with open(args.model_cfg, 'r') as f:
            cfg_dict = json.load(f)
        # Ensure device is set
        cfg_dict['device'] = device
        # Create a simple config namespace
        config = SimpleNamespace(**cfg_dict)
        config.vocab_size = tokenizer.vocab_size
        # Instantiate and move model to device
        model = MambaAudioModel(config).to(device)
        return model
    

def get_trainer_wandb_run(trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            return callback._wandb
    return None

def train_model(tokenized_dataset, tokenizer, args):
    """Train the GPTNeoX model"""

    # train_dataloader = DataLoader(tokenized_dataset['train'], collate_fn=collate_fn, batch_size=1)
    logger.info("Initializing model for training...")
    
    # Initialize wandb for tracking
    wandb.init(project="music-generation", name=f"{args.model_name}-{args.tokenizer_type}")
    
    # Define model configuration
    model = load_model(args.model_name, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        do_eval=True,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps", 
        eval_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        report_to="wandb",
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        save_safetensors=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss_forced",
        greater_is_better=False,
    )
    assert len(tokenized_dataset['validation']) > 0
    # Initialize trainer
    from transformers import TrainerCallback

    class ForceEvalLossCallback(TrainerCallback):
        def __init__(self, trainer, eval_dataset=None):
            super().__init__()
            
            self.eval_dataset = eval_dataset
            self.trainer_ref = trainer
            self.best_loss = float('inf')
            self.no_improvement_steps = 0
            self.early_stopping_steps = args.early_stopping_steps

        def compute_dtw_metrics(self, original_audio, reconstructed_audio, sample_rate):
            """Compute DTW metrics between original and reconstructed audio"""
            try:
                # Make sure lengths match
                min_len = min(len(original_audio), len(reconstructed_audio))
                if min_len == 0:
                    logger.warning("Empty audio sequence detected")
                    return {'dtw_distance': float('inf'), 'normalized_dtw': float('inf')}
                    
                # Convert to float32 if needed
                original_trim = original_audio[:min_len].astype(np.float32)
                reconstructed_trim = reconstructed_audio[:min_len].astype(np.float32)
                
                # Normalize audio to [-1, 1] range
                original_trim = original_trim / np.max(np.abs(original_trim))
                reconstructed_trim = reconstructed_trim / np.max(np.abs(reconstructed_trim))
                
                # Extract MFCC features
                n_mfcc = 13
                mfcc_orig = librosa.feature.mfcc(y=original_trim, sr=sample_rate, n_mfcc=n_mfcc)
                mfcc_recon = librosa.feature.mfcc(y=reconstructed_trim, sr=sample_rate, n_mfcc=n_mfcc)
                
                if mfcc_orig.size == 0 or mfcc_recon.size == 0:
                    logger.warning("Empty MFCC features detected")
                    return {'dtw_distance': float('inf'), 'normalized_dtw': float('inf')}
                
                # Transpose to time-major format
                mfcc_orig = mfcc_orig.T
                mfcc_recon = mfcc_recon.T
                
                # Calculate DTW distance
                dtw_distance, _ = fastdtw(mfcc_orig, mfcc_recon, dist=euclidean)
                normalized_dtw = dtw_distance / (len(mfcc_orig) + len(mfcc_recon))
                
                return {
                    'dtw_distance': float(dtw_distance),
                    'normalized_dtw': float(normalized_dtw)
                }
            except Exception as e:
                logger.error(f"Error computing DTW metrics: {str(e)}")
                return {'dtw_distance': float('inf'), 'normalized_dtw': float('inf')}

        def on_evaluate(self, args, state, control, **kwargs):
            if self.eval_dataset is None or self.trainer_ref is None:
                return

            # Determine if this is a test evaluation by checking for "test" in any metric key
            metrics = kwargs.get("metrics", {})
            is_test = any("test" in key for key in metrics.keys())
            metric_prefix = "test" if is_test else "eval"

            dataloader = self.trainer_ref.get_eval_dataloader(self.eval_dataset)
            model = self.trainer_ref.model
            model.eval()

            total_loss = 0
            total_samples = 0
            total_segments = 0
            total_segment_length = 0
            
            # Initialize DTW metrics
            total_dtw_distance = 0
            total_normalized_dtw = 0
            dtw_samples = 0
            dtw_failures = 0

            for batch_idx, batch in enumerate(dataloader):
                logger.debug(f"Processing batch {batch_idx}")
                
                # Move tensors to device and calculate segment lengths
                input_ids = batch["input_ids"]
                if isinstance(input_ids, list):
                    # For segmented inputs, move each tensor in the list
                    batch["input_ids"] = [t.to(device) for t in input_ids]
                    # Calculate segment lengths
                    for segment in input_ids:
                        total_segments += 1
                        total_segment_length += segment.size(-1)
                else:
                    # For regular inputs, move the tensor directly
                    batch["input_ids"] = input_ids.to(device)
                    # For non-segmented inputs, count as one segment
                    total_segments += 1
                    total_segment_length += segment_size if model_name == 'armt' else input_ids.size(-1)

                # Move other tensors to device
                for k, v in batch.items():
                    if k in ['attention_mask', 'labels']:
                        if isinstance(v, list):
                            batch[k] = [t.to(device) for t in v]
                        else:
                            batch[k] = v.to(device)

                with torch.no_grad():
                    outputs = model(**batch)
                    
                    assert 'logits' in outputs, "Model outputs must contain logits"
                    logger.debug(f"Logits shape: {outputs.logits.shape}")
                    
                    # Get the logits from the model output
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        
                        # Get the predicted tokens (argmax over vocabulary)
                        predicted_tokens = torch.argmax(logits, dim=-1)
                        logger.debug(f"Predicted tokens shape: {predicted_tokens.shape}")
                        
                        # Convert tokens to list and validate
                        token_list = predicted_tokens[0].tolist()
                        logger.debug(f"Token range: min={min(token_list)}, max={max(token_list)}")
                        
                        # Clear GPU cache before decoding
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Decode tokens back to audio
                        reconstructed_audio = tokenizer.decode_audio(token_list)
                        if reconstructed_audio is not None:
                            logger.debug("Successfully decoded audio from tokens")
                            # Get original audio from batch
                            original_audio = batch["input_ids"][0].cpu().numpy()
                            
                            # Debug print shapes
                            logger.debug(f"Original audio shape: {original_audio.shape}")
                            logger.debug(f"Reconstructed audio shape: {reconstructed_audio.shape}")
                            
                            # Compute DTW metrics
                            dtw_metrics = self.compute_dtw_metrics(
                                original_audio,
                                reconstructed_audio,
                                sample_rate
                            )
                            
                            if dtw_metrics['dtw_distance'] != float('inf'):
                                total_dtw_distance += dtw_metrics['dtw_distance']
                                total_normalized_dtw += dtw_metrics['normalized_dtw']
                                dtw_samples += 1
                                logger.debug(f"Successfully computed DTW metrics for sample {dtw_samples}")
                            else:
                                dtw_failures += 1
                                logger.warning(f"DTW computation failed for sample {dtw_samples + dtw_failures}")
                        else:
                            logger.warning("Failed to decode audio from tokens")
                            dtw_failures += 1
                    else:
                        logger.warning("Model outputs do not contain logits")
                        dtw_failures += 1

                # Clear GPU cache after processing batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                loss_tensor = outputs.get("ce_loss", outputs.loss)
                batch_size = batch["input_ids"][0].size(0)
                total_loss += loss_tensor.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples
            avg_segment_length = total_segment_length / total_segments if total_segments > 0 else 0
            
            # Calculate average DTW metrics
            avg_dtw_distance = total_dtw_distance / dtw_samples if dtw_samples > 0 else float('inf')
            avg_normalized_dtw = total_normalized_dtw / dtw_samples if dtw_samples > 0 else float('inf')
            
            print(f"Forced {metric_prefix} loss: {avg_loss}")
            print(f"Average segment length: {avg_segment_length:.2f}")
            print(f"Average DTW distance: {avg_dtw_distance:.4f}")
            print(f"Average normalized DTW: {avg_normalized_dtw:.4f}")
            print(f"DTW computation failures: {dtw_failures} out of {dtw_samples + dtw_failures} samples")
            
            # Log metrics to wandb
            get_trainer_wandb_run(self.trainer_ref).log({
                f"{metric_prefix}/loss_forced": avg_loss,
                f"{metric_prefix}/avg_segment_length": avg_segment_length,
                f"{metric_prefix}/dtw_distance": avg_dtw_distance,
                f"{metric_prefix}/normalized_dtw": avg_normalized_dtw,
                f"{metric_prefix}/dtw_failures": dtw_failures
            })

            # Inject metrics into the trainer logs
            if kwargs.get("metrics") is not None:
                kwargs["metrics"][f"{metric_prefix}_loss_forced"] = avg_loss
                kwargs["metrics"][f"{metric_prefix}_avg_segment_length"] = avg_segment_length
                kwargs["metrics"][f"{metric_prefix}_dtw_distance"] = avg_dtw_distance
                kwargs["metrics"][f"{metric_prefix}_normalized_dtw"] = avg_normalized_dtw
                kwargs["metrics"][f"{metric_prefix}_dtw_failures"] = dtw_failures

            # Early stopping logic
            if not is_test and self.early_stopping_steps is not None:
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.no_improvement_steps = 0
                else:
                    self.no_improvement_steps += 1
                    if self.no_improvement_steps >= self.early_stopping_steps:
                        print(f"Early stopping triggered after {self.no_improvement_steps} steps without improvement")
                        control.should_training_stop = True

            return control

    # Create a partial function for collate_fn with model_name
    from functools import partial
    collate_fn_with_model = partial(collate_fn, model_name=args.model_name)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=collate_fn_with_model,
    )
    trainer.add_callback(ForceEvalLossCallback(trainer, tokenized_dataset['validation']))
    
    # Train model
    logger.info("Starting model training...")
    metrics = trainer.evaluate()
    print(metrics)
    trainer.train()
    
    # Final evaluation on validation set
    logger.info("Running final evaluation on validation set...")
    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)
    
    # Evaluate on test set if available
    if 'test' in tokenized_dataset:
        logger.info("Running evaluation on test set...")
        test_metrics = trainer.evaluate(tokenized_dataset['test'], metric_key_prefix='test')
        print("Test metrics:", test_metrics)
    
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
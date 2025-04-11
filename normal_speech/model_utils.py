import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import AutoModelForCausalLM, AutoConfig
from mamba_ssm import Mamba
from transformers.integrations import WandbCallback

# Load config (if needed; you could also pass config around)
with open("./config.json", "r") as f:
    config = json.load(f)
device = 'cuda'
vocab_size = config["vocab_size"]
n_embed = config["n_embed"]
n_heads = config["n_heads"]
n_layers = config["n_layers"]
dropout = config["dropout"]
block_size = config["block_size"]

###############################################
# Model Components: FeedForward, Block, etc.
###############################################
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
        return x + self.ffn(self.ln2(x))

class MambaAudioModel(nn.Module):
    def __init__(self, vocab_size=vocab_size, n_embed=n_embed, n_heads=n_heads, n_layers=n_layers, dropout=dropout, block_size=block_size):
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
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None
        return logits, loss

###############################################
# Wrappers and Model Initializer
###############################################
class UniversalModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @property
    def device(self):
        # Return the device of the first parameter.
        return next(self.model.parameters()).device
    
    def forward(self, *args, **kwargs):
        if "input_ids" in kwargs or "labels" in kwargs:
            input_ids = kwargs.pop("input_ids", None)
            labels = kwargs.pop("labels", None)
            out = self.model(input_ids, labels)  # Pass positionally
        else:
            out = self.model(*args, **kwargs)
        if isinstance(out, tuple):
            logits, loss = out
            if loss is not None and loss.numel() != 1:
                loss = loss.mean()
            return {"logits": logits, "loss": loss}
        return out
    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    @classmethod
    def from_pretrained(cls, output_dir, model_name, config, device="cuda"):
        from transformers import AutoModelForCausalLM
        state_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=device)
            model = initialize_model(model_name, config)
            model.load_state_dict(state_dict)
            return cls(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(output_dir)
            return cls(model)

def initialize_model(model_name, config, block_size=2000, n_segments=2):
    if model_name == 'mamba':
        from model_utils import MambaAudioModel  # if same file; adjust import if needed
        model = MambaAudioModel()
        model = UniversalModelWrapper(model)
    elif model_name == 'gptneox':
        base_config = AutoConfig.from_pretrained('../configs/gptneox_small.json')
        base_config.vocab_size = config["vocab_size"]
        base_model = AutoModelForCausalLM.from_config(base_config)
        model = base_model  # Already has a save_pretrained method
    elif model_name == 'armt':
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
        from armt.model import AssociativeMemoryCell, AssociativeRecurrentWrapper
        base_config = AutoConfig.from_pretrained('../configs/gptneox_small.json')
        base_config.vocab_size = config["vocab_size"]
        base_model = AutoModelForCausalLM.from_config(base_config)
        mem_cell_args = {
            "num_mem_tokens": 16,
            "d_mem": 64,
            "layers_attr": "gpt_neox.layers",
        }
        rmt_config = {"segment_size": block_size // n_segments, "max_n_segments": n_segments}
        model = AssociativeRecurrentWrapper(AssociativeMemoryCell(base_model, **mem_cell_args), **rmt_config)
        model = UniversalModelWrapper(model)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model.to(device)


def get_trainer_wandb_run(trainer):
     for callback in trainer.callback_handler.callbacks:
         if isinstance(callback, WandbCallback):
             return callback._wandb
     return None

from transformers import TrainerCallback

class ForceEvalLossCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset=None):
        super().__init__()
        
        self.eval_dataset = eval_dataset
        self.trainer_ref = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_dataset is None or self.trainer_ref is None:
            return

        dataloader = self.trainer_ref.get_eval_dataloader(self.eval_dataset)
        model = self.trainer_ref.model
        model.eval()

        total_loss = 0
        total_samples = 0

        for batch in dataloader:
            device = self.trainer_ref.model.device
            for k, v in batch.items():
                batch[k] = v.to(device)

            with torch.no_grad():
                outputs = model(**batch)
            loss_tensor = outputs.get("ce_loss", outputs.get("loss"))# for gptneox and armt outputs.loss)
            batch_size = batch["input_ids"].size(0)
            total_loss += loss_tensor.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        print(f"Forced eval loss: {avg_loss}")
        get_trainer_wandb_run(self.trainer_ref).log({"eval/loss_forced": avg_loss})

        # Inject it into the trainer logs so it shows up in the metrics
        if kwargs.get("metrics") is not None:
            kwargs["metrics"]["eval_loss_forced"] = avg_loss

        return control
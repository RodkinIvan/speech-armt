from transformers import Trainer

class CustomTrainer(Trainer):
    def _save(self, output_dir: str, _internal_call: bool = False):
        self.model.save_pretrained(output_dir)

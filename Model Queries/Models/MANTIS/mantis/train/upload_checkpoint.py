import fire
import os
import torch
from transformers import Trainer, TrainingArguments, TrainerState
from transformers.trainer import TRAINER_STATE_NAME

from pathlib import Path

def main(
    repo:str,
    checkpoint_path:str=None,
    hub_token:str=os.environ.get("HF_TOKEN"),
    upload_mode="model_only",

):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    checkpoint_steps = checkpoint_path.name.split("-")[-1]
    run_name = checkpoint_path.parent.name
    args = torch.load(checkpoint_path / "training_args.bin")

    if args.hub_model_id is not None:
        hub_model_id = args.hub_model_id + f"-{checkpoint_steps}"
    else:
        hub_model_id = f"{repo}/{run_name}-{checkpoint_steps}"

    if upload_mode == "model_only":
        from mantis.models.mfuyu.processor import MFuyuProcessor
        from mantis.models.mfuyu.modeling_mfuyu import MFuyuForCausalLM
        from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
        # print("Loading model from", checkpoint_path)
        if "mfuyu" in str(checkpoint_path):
            print("Loading MFuyu model")
            processor = MFuyuProcessor.from_pretrained(str(checkpoint_path))
            model = MFuyuForCausalLM.from_pretrained(str(checkpoint_path))
        elif "llava" in str(checkpoint_path):
            print("Loading MLlava model")
            processor = MLlavaProcessor.from_pretrained(str(checkpoint_path))
            model = LlavaForConditionalGeneration.from_pretrained(str(checkpoint_path))
            
        model.push_to_hub(
            hub_model_id,
            token=hub_token or args.hub_token,
            commit_message=f"Upload {hub_model_id} to Hugging Face Hub Manually"
        )
        processor.push_to_hub(
            hub_model_id,
            token=hub_token,
            commit_message=f"Upload {hub_model_id} to Hugging Face Hub Manually")
    elif upload_mode == "all_checkpoint":
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=hub_model_id,
            repo_type="model"
        )
    
if __name__ == "__main__":
    fire.Fire(main)
import gc

import argparse

import torch

from fastchat import model
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_elites", type=float, default=0.05)
    parser.add_argument("--crossover", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--mutation", type=float, default=0.01)
    parser.add_argument("--init_prompt_path", type=str, default="./assets/autodan_initial_prompt.txt")
    parser.add_argument("--dataset_path", type=str, default="./data/advbench/harmful_behaviors.csv")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--save_suffix", type=str, default="normal")
    parser.add_argument("--API_key", type=str, default=None)

    args = parser.parse_args()
    return args


def get_developer(model_name):
    developer_dict = {
        "llama2": "Meta", 
        "vicuna": "LMSYS",
        "guanaco": "TheBlokeAI", 
        "WizardLM": "WizardLM",
        "mpt-chat": "MosaicML", 
        "mpt-instruct": "MosaicML", 
        "falcon": "TII"
    }
    return developer_dict[model_name]

def load_model_and_tokenizer(
    model_path, 
    tokenizer_path=None, 
    device='cuda:0', 
    **kwargs
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_conversation_template(template_name):
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
        conv_template.system = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
    return conv_template


def add_ranks(sorted_indices):
    ranked_indices = [(index, frequency, rank+1) for rank, (index, frequency) in enumerate(sorted(sorted_indices, key=lambda x: x[1]))]
    
    return ranked_indices

def rank_indices(tensor):
    flat_tensor = tensor.flatten()
    counter = Counter(flat_tensor)
    
    sorted_indices = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    return add_ranks(sorted_indices)

def forward(
    *, 
    model, 
    input_ids, 
    attention_mask=None, 
    batch_size=512, 
    target=None
):
    """
    Run the model forward pass in batches.

    Args:
        model: The model to use for inference.
        input_ids (torch.Tensor): Tensor of token IDs (shape: [N, seq_len]).
        attention_mask (torch.Tensor, optional): Attention mask tensor.
        batch_size (int, optional): Number of samples per batch. Default is 512.
        target: Optional labels for loss calculation (not used yet).

    Returns:
        torch.Tensor: Concatenated logits from all batches.
    """
    logits_list = []
    n_samples = input_ids.size(0)

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_input_ids = input_ids[start:end]
        batch_attention_mask = (
            attention_mask[start:end] if attention_mask is not None else None
        )
        
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            use_cache=True
        )
        logits_list.append(outputs.logits)

        del batch_input_ids, batch_attention_mask, outputs
        gc.collect()

    return torch.cat(logits_list, dim=0)


def _forward(*, model, input_ids, attention_mask, batch_size=512, target = None):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None

        outputs = model(
          input_ids=batch_input_ids,
          attention_mask=batch_attention_mask,
          past_key_values=None,
          use_cache=True,
          )
        past_key_values = outputs.past_key_values


        logits.append(model(
            input_ids=batch_input_ids, 
            past_key_values=past_key_values, 
            attention_mask=batch_attention_mask
        ).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)
"""Load the model + tokenizer once, with a sanity forward pass."""
import config  # sets HF_HOME; must come before transformers is imported

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modulo.chat import chat_prompt


def load_model_and_tokenizer(model_name=None):
    model_name = model_name or config.MODEL_NAME
    print(f"Loading {model_name} ({config.DTYPE})...")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.DTYPE]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=config.DEVICE_MAP,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded. num_hidden_layers={n_layers}, device_map={getattr(model, 'hf_device_map', 'single')}")
    return model, tokenizer


def sanity_check(model, tokenizer):
    prompt = "In one sentence, what is a steering vector?"
    messages = [{"role": "user", "content": prompt}]
    text = chat_prompt(tokenizer, messages)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    print("\n--- sanity output ---")
    print(tokenizer.decode(gen, skip_special_tokens=True))
    print("--- end ---\n")


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    sanity_check(model, tokenizer)
    # report VRAM
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {(total-free)/1e9:.2f} / {total/1e9:.2f} GB used")

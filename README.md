# Toxicator RU

This project provides detailed instructions for setting up, training, and utilizing a model designed to transform
neutral sentences on Russian language into their "toxic" counterparts. We utilize the TorchTune framework along with the
LLaMA 2 7B model, and a custom dataset converted from
the [RUSSE detox 2022 competition](https://github.com/s-nlp/russe_detox_2022) to the HuggingFace platform.

Full guide is [here](./README.full.md).

## Prerequisites

Before you begin, ensure you have Python 3.11 and Python Virtual Environment installed on your system. It's recommended
to run this project on a machine with a GPU that supports CUDA, due to the computational demands of training the model.

## Setting Up the Virtual Environment

Create and activate a virtual environment to manage dependencies:

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to use the Model

Basic example of usage:

```shell
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "evilfreelancer/llama2-7b-toxicator-ru"
DEFAULT_INSTRUCTION = "Перефразируй нетоксичный текст так, чтобы он стал токсичным, сохраняя при этом исходный смысл, орфографию и пунктуацию."
DEFAULT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

# Init model and tokenizer
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Build instruct prompt
user_message = "Великолепный полёт мысли, сразу видно, что Вы очень талантливы."
prompt = DEFAULT_TEMPLATE.format(**{"instruction": DEFAULT_INSTRUCTION, "input": user_message})

# Run model
data = tokenizer(prompt, return_tensors="pt")
data = {k: v.to(model.device) for k, v in data.items()}
output_ids = model.generate(**data, max_length=256, generation_config=generation_config)[0]
output = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output)
```

Advanced interactive example in [gen.py](./gen.py).

## Links

* https://huggingface.co/evilfreelancer/llama2-7b-toxicator-ru - LLaMA 2 7B - Toxicator RU 
* https://huggingface.co/datasets/evilfreelancer/toxicator-ru - dataset
* https://api.wandb.ai/links/evilfreelancer/33t8pqze - wandb report about training

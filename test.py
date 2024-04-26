import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# MODEL_NAME = "evilfreelancer/llama2-7b-toxicator-ru"
MODEL_NAME = "./toxicator-ru-hf"
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

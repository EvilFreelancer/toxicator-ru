import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# MODEL_NAME = "evilfreelancer/llama2-7b-toxicator-ru"
MODEL_NAME = "./toxicator-ru-hf"
DEFAULT_INSTRUCTION = "Перефразируй нетоксичный текст так, чтобы он стал токсичным, сохраняя при этом исходный смысл, орфографию и пунктуацию."
DEFAULT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, max_length=4096, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


# Init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Generating config
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

# Conversation loop
while True:
    user_message = input("User: ")

    # Skip empty messages from user
    if user_message.strip() == "":
        continue

    # Build instruct prompt
    prompt = DEFAULT_TEMPLATE.format(**{"instruction": DEFAULT_INSTRUCTION, "input": user_message})

    # Generate response
    output = generate(model=model, tokenizer=tokenizer, prompt=prompt, generation_config=generation_config)
    print("Bot:", output)
    print()
    print("==============================")
    print()

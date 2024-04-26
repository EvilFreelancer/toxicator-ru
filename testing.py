import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
import csv
from tqdm import tqdm

MODEL_NAME = "evilfreelancer/llama2-7b-toxicator-ru"
# MODEL_NAME = "./toxicator-ru-hf"
DEFAULT_INSTRUCTION = "Перефразируй нетоксичный текст так, чтобы он стал токсичным, сохраняя при этом исходный смысл, орфографию и пунктуацию."
DEFAULT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

# Init model and tokenizer
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()


def generate(input_text):
    prompt = DEFAULT_TEMPLATE.format(instruction=DEFAULT_INSTRUCTION, input=input_text)
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, max_length=256, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


# Prepare to save results
header = ["input", "output_run_1", "output_run_2", "output_run_3"]
results = [header]

# Load toxic dataset
dataset = load_dataset("evilfreelancer/toxicator-ru", split="dev[:100]")
for example in tqdm(dataset):
    outputs = [generate(example["input"]) for _ in range(len(header)-1)]
    print(">>> ", example["input"], outputs)
    results.append([example["input"]] + outputs)

# Save results to CSV
CSV_FILENAME = "./toxic_transforms.csv"
with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(results)

# Toxicator RU - Model Training with TorchTune (full guide)

This project provides detailed instructions for setting up, training, and utilizing a model designed to transform
neutral sentences on Russian language into their "toxic" counterparts. We utilize the TorchTune framework along with the
LLaMA 2 7B model, and a custom dataset converted from
the [RUSSE detox 2022 competition](https://github.com/s-nlp/russe_detox_2022) to the HuggingFace platform.

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

## Accepting Model Use Agreement

Before using the LLaMA 2 7B model, you must agree to the terms and conditions outlined by the HuggingFace. Review and
accept the agreement at the following link:

https://huggingface.co/meta-llama/Llama-2-7b-hf

## Generating "Toxicator RU" dataset

Jupyter-notebook with detailed example can be found [here](./dataset_build.ipynb).

## Downloading the Model

Download the LLaMA 2 7B model locally:

```shell
tune download meta-llama/Llama-2-7b-hf --output-dir ./Llama-2-7b-hf
```

## Configuration Setup

Copy the training configuration file suited for low-memory setups:

```shell
tune cp llama2/7B_full_low_memory ./toxicator.train.yaml
```

Modify the configuration file to change directory paths from `/tmp` to the current directory:

```shell
sed -r 's#/tmp/#./#g' -i ./toxicator.train.yaml
```

The model is trained using a dataset hosted on HuggingFace, which has been prepared from the `russe_detox_2022` project.
Here's how to set it in the configuration:

```yaml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: evilfreelancer/toxicator-ru
  template: AlpacaInstructTemplate
  split: train
  train_on_input: True
seed: null
shuffle: True
```

Card of [evilfreelancer/toxicator-ru](https://huggingface.co/datasets/evilfreelancer/toxicator-ru) dataset on
HuggingFace.

## Training the Model

If you wish to use Weights & Biases for tracking experiments, log in using the following command:

```shell
wandb login
```

Launch the training process with the configured settings:

```shell
tune run full_finetune_single_device --config toxicator.train.yaml
```

## Inference Setup

Copy the inference configuration file:

```shell
tune cp generation ./toxicator.gen.yaml
```

Modify the configuration file to change directory paths from `/tmp` to the current directory:

```shell
sed -r 's#/tmp/#./#g' -i ./toxicator.gen.yaml
```

Next need to update `checkpointer` section:

```yaml
checkpointer:
  _component_: torchtune.utils.FullModelHFCheckpointer
  checkpoint_dir: ./Llama-2-7b-hf/
  checkpoint_files: [
      hf_model_0001_2.pt,
      hf_model_0002_2.pt,
  ]
  output_dir: ./Llama-2-7b-hf/
  model_type: LLAMA2
```

As you can see `checkpoint_files` subsection was changed from defaults.

## Links

* https://huggingface.co/evilfreelancer/llama2-7b-toxicator-ru - LLaMA 2 7B - Toxicator RU 
* https://huggingface.co/datasets/evilfreelancer/toxicator-ru - dataset
* https://api.wandb.ai/links/evilfreelancer/33t8pqze - wandb report about training

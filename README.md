
Подготавливаем виртуальное окружение

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Далее принимаем соглашение:

https://huggingface.co/meta-llama/Llama-2-7b-hf

Скачиваем модель на локальный диск:

```shell
tune download meta-llama/Llama-2-7b-hf --output-dir ./Llama-2-7b-hf
```

Копируем конфигурационные файлы:

```shell
tune cp llama2/7B_full_low_memory ./toxicator.train.yaml
```

Теперь нам понадобится внести некоторые правки в конфигурационный файл, прежде всего заменим `/tmp` на `./`:

```shell
sed -r 's#/tmp/#./#g' -i ./toxicator.train.yaml
```

Используемый датасет https://github.com/s-nlp/russe_detox_2022

Он был конвертирован в https://huggingface.co/datasets/evilfreelancer/toxicator-ru

Заменяем секцию dataset на следующего вида код:

```yml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: evilfreelancer/toxicator-ru
  template: AlpacaInstructTemplate
  split: train
  train_on_input: True
seed: null
shuffle: True
```

Логинимся в W&B (опционально)

```shell
wandb login
```

Запускаем обучение:

```shell
tune run full_finetune_single_device --config toxicator.train.yaml
```

Копируем конфигурацию инференса:

```shell
tune cp generation ./toxicator.gen.yaml
```


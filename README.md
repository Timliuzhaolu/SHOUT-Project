# Listening to mental health crisis needs at scale: using Natural Language Processing to understand and evaluate a mental health crisis text messaging service. 

This repository contains Python implementation of transformer-based NLP methods for mental health text as described [here](https://www.frontiersin.org/articles/10.3389/fdgth.2021.779091/full).

## Requirements
- python 3.9.1
- numpy 1.19.4
- pandas 1.1.5
- matplotlib 3.1.3
- simpletransformers 0.51.13
- pytorch 1.7.1
- transformers 4.2.1
- tqdm 4.55.0
- nlpaug 1.1.2 
- lime 0.2.0.1
- wordcloud 1.8.1

Optional:
- wandb 0.10.12 (for logging and model tracking)

## Masked Language Model Training

The script, run_mlm.py modifies the behaviour of language model training from the `transformers` library.

Example code to run Masked Language Modeling with a Longformer model and a dataset in CSV file, similar to that used in the thesis, can be found below.

```
python run_mlm.py \
    --model_name_or_path "MaskedLM/longformer" \
    --tokenizer_name "MaskedLM/longformer" \
    --train_file "encoded_conversations.csv" \
    --validation_file "encoded_conversations.csv" \
    --do_train \
    --do_eval \
    --output_dir "/data-imperial/MaskedLM" \
    --overwrite_output_dir \
    --max_seq_length 2048  \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_total_limit 2 \
    --logging_steps 2000 \
    --save_steps 2000
```

## Preprocessing
The directory `Preprocessings/*` consists of all the data preprocessings of the Shout dataset used in the thesis. 
- `messages_labels_merge`: raw text and annotaions - merging annotations with raw conversations
- `context_messages.ipynb`: conversation stage - inclusion of local context
- `behaviour_keys.ipynb`: behaviour key - creation of one-hot encodings
- `texter_survey.ipynb`: texter demographics - preprocessing of demographics: age 13 or under; non-binary gender; autism

Also modifications to the Longformer tokenizer are specified in `customise_tokenizer.ipynb`.

## Strategy
The directory `Strategies/*` consists of different strategies implemented to improve model performance. Details can be found in `Data_augmentation.ipynb`, `NER_dataset.ipynb` and `Tree_framework.ipynb`.

## Model Training
Models can be trained using `train.py` and hyperparameter selection can be performed by `param_tuning.py`.

## Word Cloud
Word clouds for converstation stages and texter demographics can be generated in `wordcloud_generate.py` by the significant words for corresponding model detected by LIME.

## Participation Bias AC-MCC
Library developed by the former Master student Daniel Cahn can be found in `posterior.py` for applying AC-MCC to detect and reverse participation biases discussed in the report. 

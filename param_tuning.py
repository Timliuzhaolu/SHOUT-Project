import simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import wandb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('context_messages_labels_1000_1234.csv')

multiclass_df = df[['3_messages', 'convo_stage']]
multiclass_df.columns = ['text', 'labels']


train_df, test_df = train_test_split(multiclass_df, test_size = 0.20, random_state = 100)
train_df = train_df.dropna()
test_df = test_df.dropna()



sweep_config = {
    "method": "grid",  # grid, random
    "metric": {"name": "acc", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"values": [5]},
        "learning_rate": {"values": [2e-5, 1.5e-5, 2.5e-5]},
        "warmup_ratio": {"values": [0.04, 0.06, 0.09]},
        "max_seq_length": {"values": [256]},
        # "weight_decay": {"values": [0.01, 0.02]},
    },
}

sweep_id = wandb.sweep(sweep_config, project = "param_selection_3messages")
model_args = ClassificationArgs(save_model_every_epoch = False,
                                eval_batch_size = 4,
                                train_batch_size = 4,
                                use_early_stopping = True,
                                overwrite_output_dir = True,
                                output_dir = "multiclass_cls/",
                                cache_dir = "multiclass_cls/cache_dir/",
                                early_stopping_delta = 0.01,
                                early_stopping_metric = "acc",
                                early_stopping_metric_minimize = False,
                               )

def train():
    wandb.init()
    # Create a ClassificationModel
    model = ClassificationModel(
        'longformer',
        'MaskedLM/checkpoint_240000',
        num_labels = 6,
        args=model_args,
        sweep_config=wandb.config

    ) 
        # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc = accuracy_score)
    
    wandb.log({"acc": result['acc']})
    # Sync wandb
    wandb.join()


wandb.agent(sweep_id, train)
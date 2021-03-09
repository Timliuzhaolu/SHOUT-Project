import simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import wandb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('context_messages_labels_1000_1234.csv')

train_data = df[['encoded_message', 'convo_stage']]
train_data.columns = ['text', 'labels']

seed = 42
n=5
kf = KFold(n_splits=n, random_state=seed, shuffle=True)

results = []

for train_index, val_index in kf.split(train_data):
        # splitting Dataframe (dataset not included)
    train_df = train_data.iloc[train_index]
    val_df = train_data.iloc[val_index]
    # Defining Model
    model_args = ClassificationArgs(num_train_epochs=1, 
                                    overwrite_output_dir = True,
                                    output_dir = "multiclass_cls/",
                                    max_seq_length = 128,
                                    eval_batch_size = 2,
                                    train_batch_size = 2,
                                    cache_dir = "multiclass_cls/cache_dir/"
                                   )

    # Create a ClassificationModel
    model = ClassificationModel(
        'longformer',
        'MaskedLM/checkpoint_80000',
        num_labels=6,
        args=model_args
    ) 
    model.train_model(train_df)
        # validate the model
    result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=accuracy_score)
    print(result['acc'])
        # append model score
    results.append(result['acc'])


print("results",results)
print(f"Mean-ACC: {sum(results) / len(results)}")
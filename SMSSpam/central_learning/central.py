import numpy as np
import os
from tqdm import trange
from tqdm import tqdm as tqdm
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error()

parser = argparse.ArgumentParser(description='Config.')
parser.add_argument('--client_path', type = str, default = '../data/test_clients')
parser.add_argument('--test_path', type = str, default = '../data/test.csv')
parser.add_argument('--train_epoch', type = int, default = 10)
args = parser.parse_args()
datapath = args.client_path
testpath = args.test_path
trainepoch = args.train_epoch

MAX_SEQ_LENGTH=80

# Class for inputs.
class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_samples_to_inputs(sample_texts, sample_labels, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    input_items = []
    samples = zip(sample_texts, sample_labels)
    for (text, label) in tqdm(samples):
        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]", max_seq_length=512)
        
        """Padding."""
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
        
        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return input_items


def get_data_loader(features, batch_size=32, shuffle=True): 
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, device):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluation iteration"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)[:2]
       
        outputs = np.argmax(logits.to("cpu"), axis=1)
        label_ids = label_ids.to("cpu").numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    predicted_labels = np.array(predicted_labels)
    correct_labels = np.array(correct_labels)

    TP, TN, FN, FP = 0.000, 0.000, 0.000, 0.000
    TP += ((predicted_labels == 1) & (correct_labels == 1)).sum()
    TN += ((predicted_labels == 0) & (correct_labels == 0)).sum()
    FN += ((predicted_labels == 0) & (correct_labels == 1)).sum()
    FP += ((predicted_labels == 1) & (correct_labels == 0)).sum()
    p, r, F1 = 0, 0, 0
    if (TP + FP) != 0:
        p = TP / (TP + FP)
    if (TP + FN) != 0:
        r = TP / (TP + FN)
    if (r + p) != 0:
        F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
        
    return eval_loss, p, r, F1, acc


def train(model, train_dataloader, dev_dataloader, device,output_model_file="./bert.bin",
          num_train_epochs=5, patience=2, gradient_accumulation_steps=1, max_grad_norm=5,
          warmup_proportion=0.1, batch_size=32, learning_rate=5e-5): 
    
    num_train_steps = int(58475 / batch_size / gradient_accumulation_steps * num_train_epochs)
    num_warmup_steps = int(warmup_proportion * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps = num_warmup_steps)

    dev_loss, p, r, F1, acc = evaluate(model, dev_dataloader, device=device)
    history = {
        'loss': [dev_loss],
        'precision': [p],
        'recall': [r],
        'accuracy': [acc],
        'F1_score': [F1]
    }
    no_improvement = 0
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                optimizer.step()
                optimizer.zero_grad() 
                scheduler.step()

        dev_loss, p, r, F1, acc = evaluate(model, dev_dataloader, device=device)

        print("Dev loss:", dev_loss)
        print("Dev acc:", acc)

        ### You can enable the patience settings.
        # if len(loss_history) == 0 or dev_loss < min(loss_history):
        #     no_improvement = 0
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     torch.save(model_to_save.state_dict(), output_model_file)
        # else:
        #     no_improvement += 1
        
        # if no_improvement >= patience:
        #     print("No improvement on development set. Finish training.")
        #     break
        ###

        history['loss'].append(dev_loss)
        history['precision'].append(p)
        history['recall'].append(r)
        history['accuracy'].append(acc)
        history['F1_score'].append(F1)
        print("History of metrics:", history)
        
    return output_model_file

# Define the model.
BERT_MODEL = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
model.to(device)

client = pd.DataFrame()
for i in range(len(os.listdir(datapath))):   
    da = pd.read_csv(os.path.join(datapath,'client{}.csv'.format(i)))
    if len(da) == 0:
        print(i)
    client = pd.concat([client, da], ignore_index=True)

## Print XX
langdata = client
print(len(langdata))

texts = [str(i).strip().strip("\n") for i in langdata['text']]
labels = [j for j in langdata['label']]

test_langdata = pd.read_csv(testpath)

test_texts = [str(i).strip().strip("\n") for i in test_langdata['text']]
test_labels = [j for j in test_langdata['label']]

# rest_texts, test_texts, rest_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=21)
# train_texts, dev_texts, train_labels, dev_labels = train_test_split(rest_texts, rest_labels, test_size=0.25, random_state=1)

data = {"train": {}, "dev": {}, "test": {}}

data["train"]["texts"] = texts
data["dev"]["texts"] = test_texts
# data["test"]["texts"] = test_texts

data["train"]["labels"] = labels
data["dev"]["labels"] = test_labels
# data["test"]["labels"] = test_labels

for c in ["train", "dev", "test"]:
    if len(data[c]) == 0:
        continue
    print(f"{c}: {len(data[c]['texts'])} ")

train_features = convert_samples_to_inputs(data["train"]["texts"], 
                                            data["train"]["labels"], 
                                            MAX_SEQ_LENGTH, tokenizer)

dev_features = convert_samples_to_inputs(data["dev"]["texts"], 
                                          data["dev"]["labels"], 
                                          MAX_SEQ_LENGTH, tokenizer)

# test_features = convert_samples_to_inputs(data["test"]["texts"], 
#                                            data["test"]["labels"], 
#                                            MAX_SEQ_LENGTH, tokenizer)

# data["test"]["dataloader"] = get_data_loader(test_features, shuffle=False)

dataloader_train_all = get_data_loader(train_features, shuffle=True)
dataloader_dev_all = get_data_loader(dev_features, shuffle=False)

model_file_name = train(model, dataloader_train_all, dataloader_dev_all, device, gradient_accumulation_steps=4, num_train_epochs=trainepoch, 
                        output_model_file="./bert_test.bin")

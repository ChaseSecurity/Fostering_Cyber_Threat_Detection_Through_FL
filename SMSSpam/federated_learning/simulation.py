import numpy as np
import os
import pandas as pd
import json
from tqdm import trange
from tqdm.notebook import tqdm
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import importlib
import datetime
import argparse
import random

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
from flwr.common import Metrics

import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Dict, List, Optional, Tuple

## Need configuration.
client_num_cpus = 4
learningRate = 5e-5

parser = argparse.ArgumentParser(description='Config.')
parser.add_argument('--gpu', type=float, default=0)
parser.add_argument('--client_path', type=str, default='../data/test_clients')
parser.add_argument('--evaluate_path', type=str, default='../data/test.csv')
parser.add_argument('--client_num', type=int, default=3)
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--client_epoch', type=int, default=2)
parser.add_argument('--rounds', type=int, default=3)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--client_min', type=int, default=3)
parser.add_argument('--agr', type=str, default='FedAvg')
parser.add_argument('--model_path',type=str,default='no')
parser.add_argument('--compromised_num', type=int,default=0)
parser.add_argument('--p', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0)

args = parser.parse_args()
client_num_gpus = args.gpu
client_num = args.client_num
client_min = args.client_min
ratio = args.ratio
client_epoch = args.client_epoch
rounds = args.rounds
evaluate_path = args.evaluate_path
client_path = args.client_path
batch_size = args.batch_size
agr=args.agr
model_path = args.model_path

compromised_num = args.compromised_num
p = args.p
beta = args.beta # For Trimmed Mean

#  Load the AGR.
AGR = getattr(importlib.import_module(f"flwr.server.strategy.{agr.lower()}"), agr)

# Randomize the compromised clients.
a = range(1, client_num+1)
random.seed(42)
compromised = random.sample(a,compromised_num)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
print(DEVICE)

BERT_MODEL = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
Net = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)

MAX_SEQ_LENGTH=80

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):
        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]", max_length = 512)
        
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
    data2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data2, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, device):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    model.to(device)
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids, attention_mask=input_mask,token_type_ids=segment_ids, labels=label_ids)[:2]
       
        outputs = np.argmax(logits.to("cpu"), axis=1)
        label_ids = label_ids.to("cpu").numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)

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
        
    return eval_loss, np.mean(predicted_labels == correct_labels), p, r, F1, acc


def train(model, train_dataloader, device,
          num_train_epochs,  learning_rate, gradient_accumulation_steps=1, max_grad_norm=5,
          warmup_proportion=0.1): 
    
    num_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * num_train_epochs)
    num_warmup_steps = int(warmup_proportion * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-08)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps = num_warmup_steps)
    
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


# Define Flower client
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    #@profile
    def __init__(self,cid, net, trainloader, valloader) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    #@profile
    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    #@profile
    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, DEVICE, client_epoch, learningRate)
        print(self.cid)
        return get_parameters(self.net), len(self.trainloader), {}
    #@profile
    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy , p, r, F1, acc = evaluate(self.net, self.valloader,DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net.to(DEVICE)
    c_path = client_path
    langdata = pd.read_csv(os.path.join(c_path, 'client{}.csv'.format(cid)), lineterminator='\n')
    
    langdata['label'] = langdata['label'].astype(int)
    langdata = langdata.dropna()

    if eval(cid) not in compromised:
        texts = [i.strip().strip("\n") for i in langdata['text']]
        labels = [j for j in langdata['label']]
    else:
        dynamic = langdata.sample(frac=p, random_state=21)
        remain = langdata[~langdata.index.isin(dynamic.index)]
        texts = []
        for i in dynamic['text']:
            texts.append(i.strip().strip("\n"))
        for i in remain['text']:
            texts.append(i.strip().strip("\n"))

        labels = []
        for i in dynamic['label']:
            labels.append(1-i)
        for i in remain['label']:
            labels.append(i)


    langdata2 = pd.read_csv(os.path.join(c_path, 'client{}.csv'.format(cid)), lineterminator='\n')
    langdata2['label'] = langdata2['label'].astype(int)
    langdata2 = langdata2.dropna()
    texts2 = [i.strip().strip("\n") for i in langdata2['text']]
    labels2 = [j for j in langdata2['label']]

    data = {"train": {}, "dev": {}}

    data["train"]["texts"] = texts
    data["dev"]["texts"] = texts2

    data["train"]["labels"] = labels
    data["dev"]["labels"] = labels2

    train_features = convert_examples_to_inputs(data["train"]["texts"], 
                                                data["train"]["labels"], 
                                                MAX_SEQ_LENGTH, tokenizer)

    dev_features = convert_examples_to_inputs(data["dev"]["texts"], 
                                              data["dev"]["labels"], 
                                              MAX_SEQ_LENGTH, tokenizer)
    
    trainloader = get_data_loader(train_features, MAX_SEQ_LENGTH,shuffle=True,batch_size=batch_size)

    valloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, shuffle=False,batch_size=batch_size)
    
    # Create a  single Flower client representing a single organization
    return FlowerClient(cid, net, trainloader, valloader)


class SaveModelStrategy(AGR):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) :
        """Aggregate model weights using weighted average and store checkpoint"""
        net = Net.to(DEVICE)
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        print(f"Saving round {server_round} aggregated_parameters...")

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            if model_path != 'no':
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(net.state_dict(), f"{model_path}/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net.to(DEVICE)
    # valloader = testloaders[10]
    
    langdata = pd.read_csv("{}".format(evaluate_path), lineterminator='\n')
    langdata['label'] = langdata['label'].astype(int)
    texts = [i.strip().strip("\n") for i in langdata['text']]
    labels = [j for j in langdata['label']]


    test_features = convert_examples_to_inputs(texts, 
                                                labels, 
                                                MAX_SEQ_LENGTH, tokenizer)


    valloader = get_data_loader(test_features, MAX_SEQ_LENGTH, shuffle=False,batch_size=batch_size)

    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy , precision, recall, F1_score, acc= evaluate(net, valloader, DEVICE)

    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}/ precision {float(precision)} / recall {float(recall)} / F1_score {float(F1_score)} / acc {float(acc)}")
    return loss, {"accuracy": accuracy,"precision":float(precision), "recall":float(recall), "F1_score":float(F1_score), "acc": float(acc)}


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


### Define strategy
strategy_params = {
    'fraction_fit': ratio,
    'fraction_evaluate': 0,
    'min_fit_clients': client_min,
    'min_evaluate_clients': 0,
    'min_available_clients': client_num,
    'evaluate_fn': evaluate,
}
if agr == 'Krum':
    strategy_params['num_malicious_clients'] = compromised_num
    strategy_params['num_clients_to_keep'] = client_min - compromised_num
elif agr == 'FedTrimmedAvg':
    strategy_params['beta'] = beta
    
# Change 'AGR' to 'SaveModelStrategy' can save the model
strategy = AGR(**strategy_params)

metrics = fl.simulation.start_simulation(
    client_fn = client_fn,
    num_clients=client_num,
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=strategy,
    client_resources={
        "num_cpus": client_num_cpus,
        "num_gpus": client_num_gpus,
    },
)

# Write results to json files.
def write_metrics_json_file(path):
    metrics_name = ['loss','precision','recall','F1_score', 'accuracy']
    if not os.path.exists(path):
        os.mkdir(path)
    # Write losses to json file.
    for metric_name in metrics_name:
        json_path = f"{path}/{client_path}_{datetime.datetime.now().strftime('%m_%d_%H')}_{metric_name}.json"
        with open(json_path,'w') as f:
            if metric_name == 'loss':
                list_name = metrics.losses_centralized
            else:
                list_name = metrics.metrics_centralized[metric_name]
            for data in list_name:
                line = json.dumps(data)
                f.write(line)
                f.write("\n")

write_metrics_json_file('./logs')

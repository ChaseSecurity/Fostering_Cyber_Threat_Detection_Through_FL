
# Federated Learning Based SMS Spam Detection

To quickly run FL experiments,  the default settings can be adopted, by following the instructions as below.


### Default Settings

> You can find the requirements.txt in [requirements.txt](../requirements.txt)

- Python version: `3.8.10`
- Client num: 3
- Client selected each round: 3
- Resource: if gpu is available, choose gpu, otherwise cpu
- Client dataset path: `../data/test_clients` (Please split the dataset before testing.)

Attacking settings:

- Compromised clients num: 0
- Poisoning rate: 1.0
- Beta for Trimmed Mean: 0

```bash
python3 simulation.py --client_path '../data/test_clients' --rounds 5 --ratio 1 --client_num 3 >> log_3_3.txt
```

> The training tasks in our research were configured with 20 and 200 clients for over 200 rounds. More details can be found in our paper.

### Arguments

- `client_path` refers to the client datasets path.
- `ratio` refers to how much clients are chosen each round.
- `client_num` refers to the client number in total.
- `rounds` refers to the federated learning rounds.
- `gpu` GPU numbers.

Attacking arguments:

- `compromised_num` compromised clients num for attacking, the M in the paper.
- `p` poisoning rate for each compromised clients for attacking, the p in the paper.
- `beta` the argument for Trimmed Mean AGR.

More arguments:

- `client_epoch` the local training epoch for each client. Default as 2.
- `batch_size` the local training batch size for each client. Default as 2.
- `agr` the aggregation rule or stragegy. Default as FedAVG.
- `model_path` initialized model for training. Default as no.

### Aggregation rules

If you want to try more aggregation rules as listed in our paper, you can find them in [../../strategy](../../strategy).

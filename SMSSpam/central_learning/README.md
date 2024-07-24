# Central Training Based SMS Spam Detection

To get a quick start with limited resources, the default settings can be ran by following the instrucrions but without ideal performance.

### Default Settings

> You can find the requirements.txt in [requirements.txt](../requirements.txt)

- Python version: `3.8.10`
- Train dataset path: `../data/test_clients` 
- Test path: `../data/test.csv`
- Epochs to run: 20

```bash
if [ ! -d './logs' ]; then
  mkdir ./logs
fi
python3 central.py --client_path '../data/test_clients' --test_path '../data/test.csv' --train_epoch 3 >> ./logs/log_$(date +%m%d%H)
```

### Arguments

- `client_path` refers to the train dataset path.
- `test_path` test_dataset
- `train_epoch` training epochs.

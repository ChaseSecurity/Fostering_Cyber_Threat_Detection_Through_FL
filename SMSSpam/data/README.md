# 1. Get your dataset ready.

For SMS spam detection, the datasets are stored in the `csv` files.

## Central learning

- This **merged.csv** comes from the open source code for the paper [Clues in Tweets: Twitter-Guided Discovery and Analysis of SMS Spam](https://xianghang.me/files/ccs2022_bulk_sms.pdf)[dataset](https://github.com/opmusic/SpamHunter_dataset). 

- The **output.csv** comes from the part of twitter archive (https://archive.org/details/twitterarchive).

- The **all.csv** denotes the merge of **merged.csv** and **output.csv**.

- We further split the **exampleTrain.csv** as train dataset and **exampleTest.csv** to get started following the code of [splitClientsData.ipynb](../data_preprocessing/splitClientsData.ipynb).

## Federated learning

The [test_clients](test_clients) is prepared to simply try the federated learning. If you want to customize your clients, you can find it in [preprocessing](../data_preprocessing).





 


# GADAormer

This is the code of the paper GADAormer: Fraud Detection with Data Augmentation against Heterophily and Class Imbalance in Graphs.

## Abstract
Graph Neural Networks (GNNs) have emerged as powerful tools for fraud detection tasks. However, challenges such as class imbalance and heterophily persist in graph-based fraud detection methods. While existing GNN-based approaches often address only one of these challenges, we present GADAormer, an ultimate solution that combats both issues simultaneously. Our novel framework employs Group-level Aggregation with Data Augmentation (GADA) combined with a transformer architecture, termed GADAormer, to enhance fraud detection accuracy.

## Repository Structure
### Dataset
GADAormer is evaluated on the YelpChi and Amazon datasets, which can be downloaded from [here]([[https://github.com](https://github.com/YingtongDou/CARE-GNN)]())

### How to Run
To utilize GADAormer on datasets, users should follow these steps:
1. Download and unzip the required datasets into the `/preprocessing/` folder.
2. Use unsupervised learning techniques to obtain better embeddings, such as [HDGI](https://github.com/YuxiangRen/HDGI/tree/master) or any other suitable method.
3. Install necessary dependencies using `requirements.txt`.
4. Pre-process the data using `dataset_split.py` `graph2seq_mp.py` .
5. Execute the training script `main_transformer.py` to run GADAormer .


# Running Vertical Federated Learning Algorithm

The file `vfl_without_blockchain.py` tests the functionality of the VFL algorithm without interaction with blockchain.

### Dependencies

Running the algorithm requires the following dependencies, which can be installed via `pip`:

1. pytorch
2. numpy

### Dataset

We use an image dataset with binary classification that predicts if a medical image has Covid or not.
The dataset is vertically splitted among 4 parties so that each party holds a quadrant of each image.
The split dataset can be retrieved using the Google Drive [link](https://drive.google.com/file/d/1LUGy0TA03C-wcLBk8YGDeVJ42u2yHmY_/view?usp=sharing).

### Usage

The following command is used to run VFL code.
```bash
python vfl_without_blockchain.py <path-to-dataset> --theta <theta>
```
where `<path-to-dataset>` is the path to where the SplitCovid19 dataset is located, and `theta` is the differential privacy noise in range [0, 0.25].

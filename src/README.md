# Running Vertical Federated Learning Algorithm

The file `vfl_without_blockchain.py` tests the functionality of the VFL algorithm without interaction with blockchain.

### Dependencies

Running the algorithm requires the following dependencies, which can be installed via `pip`:

1. pytorch
2. numpy

All the code in src requires the following dependencies:
1. web3
2. eth-account
3. eth-tester
4. py-solc-x
5. py-evm
6. numpy
7. torch
8. torchvision
9. Pillow
10. tdqm
11. scikit-learn
12. matplotlib
13. argparse

Run the 'installDependencies' script to automatically installl all the necessary dependencies.

- For UNIX systems, run ```bash installDependencies.cmd```
- For Windows systems, run ```bash installDependencies.sh```

### Dataset

We use an image dataset with binary classification that predicts if a medical image has Covid or not.
The dataset is vertically splitted among 4 parties so that each party holds a quadrant of each image.
The split dataset can be retrieved using the Google Drive [link](https://drive.google.com/file/d/1LUGy0TA03C-wcLBk8YGDeVJ42u2yHmY_/view?usp=sharing).

### Usage

The following command is used to run VFL code without blockchain.
```bash
python vfl_without_blockchain.py <path-to-dataset> --theta <theta>
```
where `<path-to-dataset>` is the path to where the SplitCovid19 dataset is located, and `theta` is the differential privacy noise in range [0, 0.25].

# Running with blockchain

Execute each cell in the demo.ipynb notebook.

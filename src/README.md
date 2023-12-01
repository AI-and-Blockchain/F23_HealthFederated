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

- For UNIX systems, run ```installDependencies.cmd```
- For Windows systems, run ```installDependencies.sh```

### Dataset

We use an image dataset with binary classification that predicts if a medical image has Covid or not.
The dataset is vertically splitted among 4 parties so that each party holds a quadrant of each image.
The split dataset can be retrieved using the Google Drive [link](https://drive.google.com/file/d/1LUGy0TA03C-wcLBk8YGDeVJ42u2yHmY_/view?usp=sharing).

### Usage

The following command is used to run VFL code with/without blockchain.
```bash
python demo.py --datapath <path-to-dataset> --datasize <dataset-size> --theta <theta> --withblockchain <use-blockchain>
```

Arguments:

1. datapath - path to the dataset
   - Default = "./"
2. datasize - portion of dataset to use. Must be 0.25, 0.5, 1.0
  - Default = 1.0
4. theta - Noise value (in range [0, 0.25])
  - Default = 0.1
5. withblockchain - whether to use blockchain or not
  - Default = False



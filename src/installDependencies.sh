#!/bin/bash
echo
echo "Installing dependencies..."
echo

# integration dependencies
pip install web3
pip install eth-account
pip install eth-tester
pip install py-solc-x
pip install py-evm
# AI dependencies
pip install numpy
pip install torch
pip install torchvision
pip install Pillow
pip install tqdm
pip install scikit-learn
pip install matplotlib
pip install argparse

echo
echo "DONE"
echo

read -p "Press any key to continue..."
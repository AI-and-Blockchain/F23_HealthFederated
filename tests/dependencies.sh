#!/bin/bash
echo
echo "Installing dependencies..."
echo

pip install web3 eth-account eth-tester py-solc-x py-evm

echo
echo "DONE"
echo

read -p "Press any key to continue..."
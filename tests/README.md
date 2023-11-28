# Running Tests

The file `test_aggregator.py` contains unit tests to test the functionality of the Aggregator smart contract. To run the test, follow the steps below.

## 1. Install Python/Pip

Make sure that [pip](https://pypi.org/project/pip/) and [python3](https://www.python.org/downloads/) are installed

To check if python is installed, type ```python --version``` in the terminal. If that does not work, you may need to try ```python3 --version```. If neither work, then you need to install python. 

To check if pip is installed, type ```pip --version``` in the terminal. If that does not work, you may need to try ```pip3 --version```. If neither work, then you need to install pip. 

## 2. Install the other Dependancies

Run the dependencies script to install the other required dependencies via pip. 

Dependencies include:

1. web3 
2. eth-account 
3. eth-tester
4. py-solc-x 
5. py-evm

## 3. Run the test script

Run the runTest script to begin execution of the unit tests



# Important notes about the python test script

1. The script compiles the aggregator contract from the relative path `..\\src\\Aggregator.sol`. Make sure the contract has the right name and is in the correct directory.
    - see constant `CONTRACT_SOURCE` in `test_aggregator.py`

2. The contract directly references the solc binary. The relative path for binary version 0.8.23 is `.\\solc-0.8.23\\solc.exe`. 
    - see constant `SOLC_BINARY_PATH` in `test_aggregator.py`

3. You can change the version by downloading a different binary, but make sure to put it in a different folder with the name `solc-<version number>`, and update any references in the code to the new binary path. 
    - change constant `SOLC_BINARY_PATH` to the new path, and make sure naming is consistent

4. Make sure the solc binary is named solc (solc.exe), anything else will not work. 


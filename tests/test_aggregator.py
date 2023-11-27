# Imports
from solcx import compile_source
from web3 import Web3
from eth_tester import EthereumTester, PyEVMBackend
from web3.providers.eth_tester import EthereumTesterProvider
import os

SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests"+ os.sep + "solc-0.8.23" + os.sep +"solc.exe"
CONTRACT_SOURCE = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "src"+ os.sep + "Aggregator.sol"

# compile contract into bytecode
def compile_contract():
    with open(CONTRACT_SOURCE, 'r') as file:
        contract_source_code = file.read()

    compiled_sol = compile_source(source=contract_source_code, solc_binary=SOLC_BINARY_PATH)
    contract_id, contract_interface = compiled_sol.popitem()
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']
    return bytecode, abi

# setup web3 connection with pyevmbackend
def setup_web3():
    eth_tester = EthereumTester(backend=PyEVMBackend())
    provider = EthereumTesterProvider(eth_tester)
    web3 = Web3(provider)
    return web3, eth_tester

# deploy compiled contract
def deploy_contract(web3, bytecode, abi):
    contract = web3.eth.contract(abi=abi, bytecode=bytecode)

    # Assuming you are using the first account from eth_tester
    from_account = web3.eth.accounts[0]

    # Deploy the contract
    tx_hash = contract.constructor().transact({'from': from_account})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return web3.eth.contract(
        address=tx_receipt.contractAddress,
        abi=abi
    )


# --------------- TEST CASES --------------- #


def test_add_participant(contract, web3):

    # Test adding a participant
    from_account = web3.eth.accounts[0]
    tx_hash = contract.functions.addParticipant().transact({'from': from_account})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)


# --------------- MAIN --------------- #


def main():
    web3, eth_tester = setup_web3()
    bytecode, abi = compile_contract()
    contract = deploy_contract(web3, bytecode, abi)

    # Call test functions
    test_add_participant(contract, web3)
    # ... call other test functions ...

    print("All tests passed")


if __name__ == "__main__":

    main()

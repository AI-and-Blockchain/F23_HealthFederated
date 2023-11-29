# Imports
from solcx import compile_source
from web3 import Web3
from eth_tester import EthereumTester, PyEVMBackend
from web3.providers.eth_tester import EthereumTesterProvider
import os
import sys

# set binary path for compilation
if sys.platform == 'win32':
    SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests" + os.sep + "solc-0.8.23-win32" + os.sep + "solc.exe"
elif sys.platform == 'darwin':
    SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests" + os.sep + "solc-0.8.23-macos" + os.sep + "solc-macos"
elif sys.platform == 'linux':
    SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests" + os.sep + "solc-0.8.23-linux" + os.sep + "solc-static-linux"
else:
    raise Exception("Unsupported OS")

CONTRACT_SOURCE = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "src"+ os.sep + "Aggregator.sol"


# --------------- SETUP FUNCTIONS --------------- #


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
# these tests will be run in order of appearance
# the tests build off each other

def test_aggregate_no_clients(contract, web3):
    try:
        # aggregate
        tx_hash = contract.functions.aggregate().transact({'from': web3.eth.accounts[2]})
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        assert False, "Aggregate cannot be called with 0 clients"
    except Exception as e:
        print("test_aggregate_no_clients passed")

def test_add_one_participant(contract, web3):
    # Test adding a participant
    from_account = web3.eth.accounts[1]
    tx_hash = contract.functions.addParticipant().transact({'from': from_account})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    # Verify if the participant is added
    client_count = contract.functions.getClientCount().call()
    assert client_count == 1, "Client count should be 1 after adding a participant"
    print("test_add_participant_and_verify passed")


def test_add_duplicate_accounts(contract, web3):
    # try adding a participant of an account already added in a previous test
    # this should fail
    try:
        from_account = web3.eth.accounts[1]
        tx_hash = contract.functions.addParticipant().transact({'from': from_account})
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        assert False, "Adding a duplicate address as a participant should have failed"
    except Exception as e:
        print("test_add_duplicate_accounts passed")
        

def test_add_participant_limit(contract, web3):
    # Assuming MAX_CLIENTS is 4
    for i in range(2, 6):  
        from_account = web3.eth.accounts[i]
        contract.functions.addParticipant().transact({'from': from_account})

    # Try adding one more participant, which should fail
    try:
        from_account = web3.eth.accounts[5]
        contract.functions.addParticipant().transact({'from': from_account})
        assert False, "Adding a participant beyond the MAX_CLIENTS limit should have failed"
    except Exception as e:
        print("test_add_participant_limit passed")


def test_participant_parameters_get_and_update(contract, web3):
    # Test updating participant parameters of a client already added
    from_account = web3.eth.accounts[1]  
    new_parameters = [[1, 2, 3], [4, 5, 6]]

    # update params
    tx_hash = contract.functions.updateParticipantParameters(new_parameters).transact({'from': from_account})
    web3.eth.wait_for_transaction_receipt(tx_hash)
    
    # get params from same address
    client_params = contract.functions.getClientParameters(from_account).call()

    # verify
    assert client_params == new_parameters, "Locally declared params should match the retrieved params from the blockchain"

    print("test_participant_parameters_get_and_update passed")


def test_participant_parameters_get_and_update_failure(contract, web3):
    incorrect_params = [[1, 2, 3], [4, 5, 7]]
    # get params from same address
    from_account = web3.eth.accounts[1]
    client_params = contract.functions.getClientParameters(from_account).call()
    assert client_params != incorrect_params, "These parameters should not match"

    print("test_participant_parameters_get_and_update_failure passed")


def test_aggregate_algorithm_and_getter(contract, web3):
    # add params to a second client
    params2 = [[1, 2, 3], [4, 5, 6]]
    from_account = web3.eth.accounts[2]
    
    # update params
    tx_hash = contract.functions.updateParticipantParameters(params2).transact({'from': from_account})
    web3.eth.wait_for_transaction_receipt(tx_hash)

    # aggregate
    tx_hash = contract.functions.aggregate().transact({'from': web3.eth.accounts[2]})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    # fetch aggregated weights
    agg_params = contract.functions.getAggregatedWeights().call()
    assert [[2, 4, 6], [8, 10, 12]] == agg_params, "Aggregation Algorithm Failed"

    print("test_aggregate_algorithm_and_getter passed")


def test_reinitialize_client_params(contract, web3):
    # reset the client params of clients we already set

    # first client 1
    new_params1 = [[1, 1, 1], [1, 1, 1]]
    # update params
    tx_hash = contract.functions.updateParticipantParameters(new_params1).transact({'from': web3.eth.accounts[1]})
    web3.eth.wait_for_transaction_receipt(tx_hash)
    # get client 1 params
    client1_params = contract.functions.getClientParameters(web3.eth.accounts[1]).call()
    assert new_params1 == client1_params, "New parameters for client1 do not match"

    # next client 2
    new_params2 = [[2, 2], [2, 2]]
    # update params
    tx_hash = contract.functions.updateParticipantParameters(new_params2).transact({'from': web3.eth.accounts[2]})
    web3.eth.wait_for_transaction_receipt(tx_hash)
    # get client 2 params
    client2_params = contract.functions.getClientParameters(web3.eth.accounts[2]).call()
    assert client2_params == new_params2, "New parameters for client2 do not match"

    print("test_reinitialize_client_params passed")


def test_aggregate_different_shapes(contract, web3):
    # using the newly initialized params of different sizes, test the aggregate algorithm

    # aggregate
    tx_hash = contract.functions.aggregate().transact({'from': web3.eth.accounts[2]})
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    
    # fetch aggregated weights
    agg_params = contract.functions.getAggregatedWeights().call()
    
    assert [[3, 3, 1], [3, 3, 1]] == agg_params, "Aggregation with different shapes failed"

    print("test_aggregtate_different_shapes passed")

def test_aggregationRounds(contract):
    # up to now, there's been two successful aggregation rounds
    # and one denied aggregation round (no clients existed)
    # test behavior of the aggregationRounds state variable

    agg_count = contract.functions.aggregationRounds().call()
    assert agg_count == 2, "Aggregation Count state variable not correct"

    print("test_aggregationRounds passed")

def main():
    web3, eth_tester = setup_web3()
    bytecode, abi = compile_contract()
    contract = deploy_contract(web3, bytecode, abi)

    # Call test functions

    # aggregate with no clients
    test_aggregate_no_clients(contract, web3)
    
    # adding clients
    test_add_one_participant(contract, web3)
    test_add_duplicate_accounts(contract, web3)
    test_add_participant_limit(contract, web3)

    # get and update client params 
    test_participant_parameters_get_and_update(contract, web3)
    test_participant_parameters_get_and_update_failure(contract, web3)

    # aggregate
    test_aggregate_algorithm_and_getter(contract, web3)

    # reinitialize client params
    test_reinitialize_client_params(contract, web3) 

    # test aggregate with different shapes
    test_aggregate_different_shapes(contract, web3)

    # test aggregation rounds state variable behavior
    test_aggregationRounds(contract, web3)

    # end
    print("All tests passed")


# --------------- MAIN --------------- #

if __name__ == "__main__":

    main()

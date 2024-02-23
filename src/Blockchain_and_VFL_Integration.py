from eth_account import Account
import secrets
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider
from eth_tester import PyEVMBackend
from solcx import compile_source
import sys
import os

'''
Usage:
num_clients - number of clients
contract_path - file path to aggregator source code
erc20_path - file path to reward token source code
'''

class BlockchainVFLIntegrator:
    def __init__(self, num_clients, contract_path, erc20_path):
        self.client_accounts = []
        
        # Generate test Ethereum accounts for each client/hospital with a private key.
        for _ in range(num_clients):
            priv = secrets.token_hex(32)
            private_key = "0x" + priv
            self.client_accounts.append(Account.from_key(private_key))

        self.w3 = Web3(EthereumTesterProvider(PyEVMBackend()))
        self.fund_client_accounts()

        erc20_address = self.compile_and_deploy_erc20(erc20_path)

        compiled_sol = self.compile_source_file(contract_path)
        self.contract_id, self.contract_interface = compiled_sol.popitem()
        self.contract_address = self.deploy_contract(self.contract_interface, erc20_address)

        self.aggregator = self.w3.eth.contract(address=self.contract_address,
                                               abi=self.contract_interface["abi"])
        self.add_clients_to_contract()
        
        # Owner arbitrarily set to first client
        self.owner_account = self.client_accounts[0]
    
    def add_clients_to_contract(self):
        for client_account in self.client_accounts:
            unsent_tx = self.aggregator.functions.addParticipant().build_transaction({
                "from": client_account.address,
                "nonce": self.w3.eth.get_transaction_count(client_account.address),
            })
            signed_tx = self.w3.eth.account.sign_transaction(unsent_tx, private_key=client_account.key)
        
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def fund_client_accounts(self):
        num_web3_accounts = len(self.w3.eth.accounts)
        num_clients = len(self.client_accounts)      
        
        for i in range(num_web3_accounts):
            j = i%num_clients
            
            # Fund client addresses based on corresponding
            # indices in the web3 addresses list
            web3_address = self.w3.eth.accounts[i]
            client_address = self.client_accounts[j].address
            web3_address_balance = self.w3.eth.get_balance(web3_address)
            
            self.w3.eth.send_transaction({
                "from": web3_address,
                "to": client_address,
                "gas": 21000,
                "gasPrice": self.w3.to_wei("50", "gwei"),
                "value": (3*web3_address_balance)//4
            })

    def compile_source_file(self, file_path):
        with open(file_path, 'r') as f:
            source = f.read()

        if sys.platform == 'win32':
            SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests" + os.sep + "solc-0.8.23-win32" + os.sep + "solc.exe"
        elif sys.platform == 'darwin':
            SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests" + os.sep + "solc-0.8.23-macos" + os.sep + "solc-macos"
        elif sys.platform == 'linux':
            SOLC_BINARY_PATH = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "tests" + os.sep + "solc-0.8.23-linux" + os.sep + "solc-static-linux"
        else:
            raise Exception("Unsupported OS")

        return compile_source(source, output_values=['abi','bin'], solc_binary=SOLC_BINARY_PATH)

    def deploy_contract(self, contract_interface, erc20_address=None):
        client_addresses = [client_account.address for client_account in self.client_accounts]
        if erc20_address:
             tx_hash = self.w3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bin']).constructor(erc20_address).transact()
        else: 
            tx_hash = self.w3.eth.contract(
                abi=self.contract_interface['abi'],
                bytecode=self.contract_interface['bin']).constructor().transact()

        address = self.w3.eth.get_transaction_receipt(tx_hash)['contractAddress']
        return address

    def compile_and_deploy_erc20(self, erc20_path):
        compiled_erc20 = self.compile_source_file(erc20_path)
        _, erc20_interface = compiled_erc20.popitem()
       
        tx_hash = self.w3.eth.contract(
            abi=erc20_interface['abi'],
            bytecode=erc20_interface['bin']).constructor("HealthFederatedToken", "HFD", 1000000).transact()
        
        erc20_address = self.w3.eth.get_transaction_receipt(tx_hash)['contractAddress']
        return erc20_address
    
    def update_client_weights(self, client_account, weights):
        if client_account not in self.client_accounts:
            return "Invalid client address"

        unsent_tx = self.aggregator.functions.updateParticipantParameters(weights).build_transaction({
            "from": client_account.address,
            "nonce": self.w3.eth.get_transaction_count(client_account.address),
        })
        signed_tx = self.w3.eth.account.sign_transaction(unsent_tx, private_key=client_account.key)
        
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def aggregate_weights(self):
        unsent_tx = self.aggregator.functions.aggregate().build_transaction({
            "from": self.owner_account.address,
            "nonce": self.w3.eth.get_transaction_count(self.owner_account.address),
        })
        signed_tx = self.w3.eth.account.sign_transaction(unsent_tx, private_key=self.owner_account.key)
        
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def get_aggregated_weights(self):
        # This function will be called by the server to get aggregated weights from the smart contract.
        return self.aggregator.functions.getAggregatedWeights().call()


if __name__=='__main__':
    CONTRACT_SOURCE = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "src"+ os.sep + "Aggregator.sol"
    # ERC_SOURCE = os.getcwd().split("F23_HealthFederated")[0] + "F23_HealthFederated" + os.sep + "src"+ os.sep + "HealthFederatedToken.sol"

    blockchain_vfl_integrator = BlockchainVFLIntegrator(4, CONTRACT_SOURCE)

    client_parameters = []

    for i in range(len(blockchain_vfl_integrator.client_accounts)):
        client_parameter = []
        for j in range(10):
            client_parameter.append([i+j+k for k in range(64)])
        client_parameters.append(client_parameter)

    for i in range(len(blockchain_vfl_integrator.client_accounts)):
        blockchain_vfl_integrator.update_client_weights(blockchain_vfl_integrator.client_accounts[i],
                                                        client_parameters[i])

    print("Sample weight update sent by a single client")
    print(client_parameters[0])

    blockchain_vfl_integrator.aggregate_weights()

    print("Aggregated weights")
    print(blockchain_vfl_integrator.get_aggregated_weights())


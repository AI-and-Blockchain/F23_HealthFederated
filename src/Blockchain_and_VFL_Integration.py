from eth_account import Account
import secrets
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider
from eth_tester import PyEVMBackend
from solcx import compile_source, install_solc, get_solcx_install_folder
import sys
import shutil

class BlockchainVFLIntegrator:
    def __init__(self, num_clients, contract_path):
        if sys.platform!="win32":
            shutil.rmtree(get_solcx_install_folder())
            install_solc()
        self.client_accounts = []
        
        # Generate test Ethereum accounts for each client/hospital with a private key.
        for _ in range(num_clients):
            priv = secrets.token_hex(32)
            private_key = "0x" + priv
            self.client_accounts.append(Account.from_key(private_key))

        self.w3 = Web3(EthereumTesterProvider(PyEVMBackend()))
        self.fund_client_accounts()

        compiled_sol = self.compile_source_file(contract_path)
        self.contract_id, self.contract_interface = compiled_sol.popitem()
        self.contract_address = self.deploy_contract(self.contract_interface)

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

        if sys.platform=="win32":
            return compile_source(source, output_values=['abi','bin'], binary_path="../tests/solc-0.8.23/solc.exe")
        else:
            return compile_source(source, output_values=['abi','bin'])

    def deploy_contract(self, contract_interface):
        client_addresses = [client_account.address for client_account in self.client_accounts]

        tx_hash = self.w3.eth.contract(
            abi=self.contract_interface['abi'],
            bytecode=self.contract_interface['bin']).constructor().transact()

        address = self.w3.eth.get_transaction_receipt(tx_hash)['contractAddress']
        return address
    
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
    blockchain_vfl_integrator = BlockchainVFLIntegrator(4, "./Aggregator.sol")

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


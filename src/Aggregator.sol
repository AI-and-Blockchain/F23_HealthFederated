// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address _to, uint256 _value) external returns (bool);
}

contract FederatedAggregator {
    
    // interface IERC20 {
    //     function transfer(address recipient, uint256 amount) external returns (bool);
    // }

    address owner;
    // struct to represent a client
    // contains an addr and a clients individual model params
    struct Client {
        address clientAddress;
        uint256 maxIndex;
    }

    // mapping(uint256 => uint256[]) represents the 2D array for each client
    // each client will have a client id, which is mapped to the 2D array mapping
    mapping(address => mapping(uint256 => uint256[])) clientParameters;
    

    // Struct to represent the global model (server)
    // maxIndex used for the mapping
    struct Server {
        uint256 clientCount;
        uint256 maxIndex;
    }
    // server params as a state variable (mapping can only exist in storage)
    mapping(uint256 => uint256[]) aggregatedParameters;

    Client[] public clients;
    Server public server;

    uint256 public aggregationRounds = 0;

    uint256 MAX_CLIENTS = 4;

    IERC20 public tokenContract; 
    uint256 public rewardAmount = 1 * 10 ** 18; // Amount of reward per update

    // initialize server in constructor
    constructor(address _tokenContract) {
        owner = msg.sender;
        tokenContract = IERC20(_tokenContract);
        server = Server({clientCount: 0, maxIndex: 0});
    }

    // function to create a client object and add it to the list
    function addParticipant() public {
        require(msg.sender != address(0), "Invalid client address");
        require(aggregationRounds == 0, "Communication Rounds have started. Cannot add new clients.");
        require(clients.length <= MAX_CLIENTS, "Max Client Count reached. Cannot add any more clients.");
        // check for duplicate addresses
        for (uint i = 0;i < clients.length; i++){
            if (clients[i].clientAddress == msg.sender){
                revert("This client already exists");
            }
        }

        clients.push(Client({clientAddress: msg.sender, maxIndex: 0}));
        server.clientCount++;
    }


    function getAggregatedWeights()  view public returns (uint256[][] memory) {
        uint256[][] memory params = new uint256[][](server.maxIndex);
        for(uint i = 0 ;i < server.maxIndex; i++ ) {
            params[i] = aggregatedParameters[i];
        }
        return params;
    }

    function getClientCount() view public returns (uint256) {
        return server.clientCount;
    }

    function getClientParameters(address clientAddr) public view returns (uint256[][] memory) {
        // look for client
        uint256 max_index = 0;
        for (uint i = 0;i < clients.length; i++){
            if (clients[i].clientAddress == clientAddr){
                max_index = clients[i].maxIndex;
                // return client params
                uint256[][] memory params = new uint256[][](max_index);
                for (uint j = 0; j < max_index; j++){
                    params[j] = clientParameters[clientAddr][j];
                }
                return params;
            }
        }
        revert("Client does not exist");
    }

    // function to update params for a client
    // if the message sender is a client, will update params with the provided arg
    function updateParticipantParameters(uint256[][] memory newParameters) public {
        // search for client and update maxIndex
        bool found = false;
        for (uint i = 0; i < clients.length; i++){
            if (clients[i].clientAddress == msg.sender) {
                clients[i].maxIndex = newParameters.length;
                found = true;
            }
        }
        if (found == false){
            revert("Sender is not a participant");
        }

        // update params
        for (uint i = 0;i < newParameters.length; i++){
            clientParameters[msg.sender][i] = newParameters[i];
        }   

        // assign reward
        require(tokenContract.transfer(msg.sender, rewardAmount), "Reward Transfer failed");
    }
    
    // function to aggregate the client params into the global params
    function aggregate() public {
        require(clients.length > 0, "No clients available");
        
        // clear old params
        for (uint i = 0;i < server.maxIndex; i++){
            delete aggregatedParameters[i];
        }

        // iterate over clients
        for (uint i = 0;i < clients.length; i++){
            address clientAddr = clients[i].clientAddress;
            uint256 clientMaxIndex = clients[i].maxIndex;

            // update server maxIndex if necessary
            if (clientMaxIndex > server.maxIndex){
                server.maxIndex = clientMaxIndex;
            }

            // iterate over client params
            for (uint j = 0; j < clientMaxIndex; j++){
                uint256[] memory clientParams = clientParameters[clientAddr][j];

                // initialize array if needed
                if (aggregatedParameters[j].length == 0){
                    aggregatedParameters[j] = new uint256[](clientParams.length);
                }
                
                // sum vector-wise
                for (uint k = 0;k < clientParams.length; k++){
                    if (k < aggregatedParameters[j].length){
                        aggregatedParameters[j][k] += clientParams[k];
                    }
                }
            }

        }
        aggregationRounds++;
    }

}

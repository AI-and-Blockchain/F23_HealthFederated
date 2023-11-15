// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedAggregator {
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

    Client[] clients;
    Server public server;

    uint256 aggregationRounds = 0;

    uint256 MAX_CLIENTS = 4;

    // initialise server in constructor
    constructor() {
        owner = msg.sender;
        server = Server({clientCount: 0, maxIndex: 0});
    }

    // function to create a client object and add it to the list
    function addParticipant() public {
        require(msg.sender != address(0), "Invalid client address");
        require(aggregationRounds == 0, "Communication Rounds have started. Cannot add new clients.");
        require(clients.length <= MAX_CLIENTS, "Max Client Count reached. Cannot add any more clients.");
 
        clients.push(Client({clientAddress: msg.sender, maxIndex: 0}));
    }


    function getAggregatedWeights()  view public returns (uint256[][] memory) {
        uint256[][] memory params = new uint256[][](server.maxIndex);
        for(uint i = 0 ;i < server.maxIndex; i++ ) {
            params[i] = aggregatedParameters[i];
        }
        return params;
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
    }
    
    // functiont to aggregate the client params into the global params using Federated Average
    function aggregate() public {
        // require(clients.length > 0, "No clients available");
        // require(msg.sender == owner, "Only the owner can start aggregation");
        // TODO: implement new algorithm
        
    }





}
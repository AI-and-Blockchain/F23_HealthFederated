// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedAggregator {
    address owner;
    // struct to represent a client
    // contains an addr and a clients individual model params
    struct Client {
        address clientAddress;
        uint256[] modelParameters;
    }

    // Struct to represent the global model (server)
    // contains num_clients and the model params
    struct Server {
        uint256[] aggregatedParameters;
        uint256 clientCount;
    }
    
    Client[] clients;
    Server public server;
    
    // keep track of amount of times aggregation has been called
    uint256 aggregationRounds = 0;

    // Maximum Client count
    uint256 MAX_CLIENTS = 4;

    // initialise server in constructor
    constructor() {
        owner = msg.sender;
        server = Server({aggregatedParameters: new uint256[](10), clientCount: 0}); // 10 output labels for MNIST
    }

    // function to create a client object and add it to the list
    function addParticipant() public {
        require(msg.sender != address(0), "Invalid client address");
        require(aggregationRounds == 0, "Communication Rounds have started. Cannot add new clients.");
        require(clients.length <= MAX_CLIENTS, "Max Client Count reached. Cannot add any more clients.");
        clients.push(Client({clientAddress: msg.sender, modelParameters: new uint256[](10)}));
    }

    // function to update params for a client
    // if the message sender is a client, will update params with the provided arg
    function updateParticipantParameters(uint256[] memory newParameters) public {
        // check for correct length
        require(newParameters.length == server.aggregatedParameters.length, "Invalid parameter length");
        for (uint256 i = 0; i < clients.length; i++) {
            if (clients[i].clientAddress == msg.sender) {
                clients[i].modelParameters = newParameters;
                return;
            }
        }
        // client addr not found
        revert("Participant not found");
    }

    // functiont to aggregate the client params into the global params using Federated Average
    function aggregate() public {
        require(clients.length > 0, "No clients available");
        require(msg.sender == owner, "Only the owner can start aggregation");
        // perform averaging over the 10 output labels
        for (uint256 i = 0; i < server.aggregatedParameters.length; i++){
            // get the sum for each client
            uint256 sum = 0;
            for (uint256 j = 0; j < clients.length; j++) {
                sum += clients[j].modelParameters[i];
            }
            // compute and store the avg
            server.aggregatedParameters[i] = sum / clients.length;
        }
    }





}
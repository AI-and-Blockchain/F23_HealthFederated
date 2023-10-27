# F23_HealthFederated

# Motivation
Hospitals have huge amounts of data that they might not like to share openly with other hospitals. However, hospitals benefit from having a Machine Learning model that has been trained on data from other hospitals for better insights on their own data. Vertical Federated Learning is a good fit for this use case. Blockchain is used for transparency and immutability of the global weight updates. Local Differential Privacy provides protection for model weight updates that are published on the Blockchain.

# Implementation
Our project uses Vertical Federated Learning for binary classification of medical image data from different hospitals. A Smart Contract is used for aggregation (summation) of local weight updates to create the global weight update. The global weight update is stored on the Blockchain and sent to a central server. The central server trains its model on the global weight update sent by the Smart Contract and sends its global weight update back to each client.

# Model Architecture 
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Model%20Architecture1.png)

# Sequence Diagram
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Sequence_Diagram.png)

# Centralized Feature Fusion Diagram
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Centralized%20Model.png)

1. At first, three CNN pre-trained models, i.e., DenseNet169, ResNet50, and VGG19, with the pre-trained weights will be adopted.
2. We will use this model without their classification layers because we want to use these for feature extraction part only.
3. All the extracted features will be combined into a single fusion vector using a concatenate layer.
4. The combined features represent high-level functionality such as sharpening, textures, roundness, and compactness of the CXR images.
5. Finally, the combined features then feed into the central server for the training and classification purpose.

# Blockchain Component

**Implementation:** A smart contract written in Solidity capable of receiving client weight updates, aggregating them and sending the aggregation to a global model. 

**The smart contract will:**

1. Allow for clients to send their nosy weight updates

2. Receive the weights and sum them together

3. Allow for a global model to receive the concatenated weights from the client

The smart contract will not perform any training, since it will have high gas costs

**Rationale:** Allows for multiple hospitals to interface with the model without the need for a centralized authority. Blockchain will provide transparency and verifiability, and local differential privacy mechanisms will ensure privacy. 


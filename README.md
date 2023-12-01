# F23_HealthFederated

# Motivation
Hospitals have huge amounts of data that they might not like to share openly with other hospitals. However, hospitals benefit from having a Machine Learning model that has been trained on data from other hospitals for better insights on their own data. Vertical Federated Learning is a good fit for this use case. Blockchain is used for transparency and immutability of the global weight updates. Local Differential Privacy provides protection for model weight updates that are published on the Blockchain.

# How to run
[src/README.md](src/README.md)

# Implementation
Our project uses Vertical Federated Learning for binary classification of medical image data from different hospitals. A Smart Contract is used for aggregation (summation) of local training results (embedding) to train the global model. The embedding sum is stored on the Blockchain and sent to a central server. The central server trains its model on the embedding sum sent by the Smart Contract and sends the gradient back to each client.

# Model Architecture 
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Model%20Architecture1.png)

# Sequence Diagram
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Sequence_Diagram.png)

# Centralized Feature Fusion Diagram
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Centralized%20Model.png)

1. At first, two CNN pre-trained models, i.e., ResNet50, and VGG19, with the pre-trained weights will be adopted for the client model.
2. We will use this model without their classification layers because we want to use these for feature extraction part only.
3. All the extracted features will be combined into a single fusion vector (embedding) using a concatenate layer.
4. The embeddings represent high-level functionality such as sharpening, textures, roundness, and compactness of the CXR images.
5. Finally, the embeddings are summed and then fed into the central server for the training and classification purpose.

# Vertical Federated Learning Algorithm

Below is a description of our Vertical Federated Learning algorithm.

In each training round:
1. A minibatch is randomly chosen for training. The IDs of the chosen samples are shared among server and clients
2. Each client generates embeddings using their local model and private data.
3. Each client adds differential privacy noise to their embeddings.
4. Each client sends their noisy embeddings to the smart contract for aggregation.
5. Smart contract sums the noisy embedding and sends to the server.
6. Server calculates the gradient w.r.t the embedding sum and sends to parties.
7. Server calculates the gradient w.r.t the global parameters and updates the global parameters.
8. Each client calculates the gradient w.r.t their local parameters using the chain rule and updates their local parameters.


# Blockchain Component

**Implementation:** A smart contract written in Solidity capable of receiving client weight updates, aggregating them and sending the aggregation to a global model. 

**The smart contract will:**

1. Allow for clients to send their noisy weight updates

2. Receive the weights and sum them together

3. Allow for a global model to receive the aggregated weights from the clients

The smart contract will not perform any training, since it will have high gas costs

**Rationale:** Allows for multiple hospitals to interface with the model without the need for a centralized authority. Blockchain will provide transparency and verifiability, and local differential privacy mechanisms will ensure privacy. 

# Dataset

We use an image dataset with binary classification that predicts if a medical image has Covid or not. The dataset is vertically splitted among 4 parties so that each party holds a quadrant of each image. The split dataset can be retrieved using the Google Drive [link](https://drive.google.com/file/d/1LUGy0TA03C-wcLBk8YGDeVJ42u2yHmY_/view?usp=sharing).


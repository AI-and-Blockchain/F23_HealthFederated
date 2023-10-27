# F23_HealthFederated

# Motivation
Hospitals have huge amounts of data that they might not like to share openly with other hospitals. However, hospitals benefit from having a Machine Learning model that has been trained on data from other hospitals for better insights on their own data. Vertical Federated Learning is a good fit for this use case. Blockchain is used for transparency and immutability of the global weight updates.

# Implementation
Our project uses Vertical Federated Learning for binary classification of medical image data from different hospitals. A Smart Contract is used for aggregation/concatenation of local weight updates to create the global weight update. The global weight update is stored on the Blockchain and sent to a central server. The central server trains its model on the global weight update sent by the Smart Contract and sends its global weight update back to each client.

# Model Architecture 
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Model%20Architecture1.png)

# Sequence Diagram
![alt text](https://github.com/AI-and-Blockchain/F23_HealthFederated/blob/main/Proj-Checkin-02-files/Sequence_Diagram.png)



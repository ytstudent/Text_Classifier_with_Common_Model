import numpy as np
from type_transform import TransformType
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from tools import Tools
from LSTM_model import RNN

# I. Load the data & transform type
trans = TransformType()
tools = Tools()

data_all_df = trans.from_txt_to_df(r".\data\data_all.txt")
training_df = trans.from_txt_to_df(r".\data\training_data.txt")
testing_df = trans.from_txt_to_df(r".\data\testing_data.txt")

encoder = LabelEncoder()
encoder.fit(data_all_df["intention"])
print(encoder.classes_)
label_dict = {}
for i, j in enumerate(encoder.classes_):
    label_dict[j] = i
print(label_dict)

# II. Get the dictionary
info_str = " , ".join(data_all_df["information"])
_, _, dictionary, _y = tools.word_bag(info_str)
print(len(dictionary))
token_num = 60

inputs_training = []
for i in training_df["information"]:
    i_input = tools.sent2array(i, token_num)
    inputs_training.append(i_input)
inputs_training = np.array(inputs_training)
target_training = encoder.transform(training_df["intention"])

inputs_testing = []
for i in testing_df["information"]:
    i_input = tools.sent2array(i, token_num)
    inputs_testing.append(i_input)
inputs_testing = np.array(inputs_testing)
target_testing = encoder.transform(testing_df["intention"])

# III. Training the model and testing
hidden_size = 64
num_layers = 2
num_classes = 4
batch_size = 2
num_epochs = 5
learning_rate = 0.01
vocabulary_size = len(dictionary)
embed_dim = 300

model = RNN(vocabulary_size, embed_dim, hidden_size, num_layers, num_classes)

# # Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
for epoch in range(num_epochs):
    i = 0
    for X_batch, y_batch, n_batches, real_len in tools.shuffle_batch(inputs_training, target_training, batch_size,
                                                                     token_num):
        X_batch = torch.from_numpy(X_batch).float()
        y_batch = torch.from_numpy(y_batch).long()

        outputs = model(X_batch, real_len)
        loss = criterion(outputs, y_batch)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, n_batches, loss.item()))
        i += 1

# # Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch, _, real_len in tools.shuffle_batch(inputs_testing, target_testing, batch_size, token_num):
        X_batch = torch.from_numpy(X_batch).float()
        y_batch = torch.from_numpy(y_batch).long()

        outputs = model(X_batch, real_len)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    print('Test Accuracy of the model : {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), r'.\save_model\LSTM_model.ckpt')

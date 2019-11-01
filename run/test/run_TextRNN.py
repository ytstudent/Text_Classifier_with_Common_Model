import torch
from torch import nn
from model.LSTM_model import RNN
from data_processing import DataProcess

# I. Load the data & transform type
token_num = 60
processing = DataProcess(token_num)

# II. Get the dictionary
inputs_training, target_training, inputs_testing, target_testing = processing.get_data()

# III. Training the model and testing
hidden_size = 64
num_layers = 1
num_classes = 4
batch_size = 64
num_epochs = 5
learning_rate = 0.01
vocabulary_size = len(processing.dictionary)
embed_dim = 300

model = RNN(vocabulary_size, embed_dim, hidden_size, num_layers, num_classes)

# # Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # Train the model
for epoch in range(num_epochs):
    i = 0
    for X_batch, y_batch, n_batches, real_len in processing.tools.shuffle_batch(inputs_training, target_training,
                                                                                batch_size,
                                                                                token_num, real_length=True):
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
    for X_batch, y_batch, _, real_len in processing.tools.shuffle_batch(inputs_testing, target_testing, batch_size,
                                                                        token_num,
                                                                        real_length=True):
        X_batch = torch.from_numpy(X_batch).float()
        y_batch = torch.from_numpy(y_batch).long()

        outputs = model(X_batch, real_len)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    print('Test Accuracy of the model : {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), r'.\save_model\LSTM_model.ckpt')

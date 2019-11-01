from data_processing import DataProcess
from model.TEXTCNN_model import TextCNN
from training_model import TrainingModel

# I. Load the data & transform type
token_num = 60
processing = DataProcess(token_num)

# II. Get the dictionary
inputs_training, target_training, inputs_testing, target_testing = processing.get_data()

# III. Training the model and testing
num_filters = 16
filter_sizes = (2, 3, 4)
dropout = 0.5
num_classes = 4
batch_size = 64
num_epochs = 10
learning_rate = 0.01
vocabulary_size = len(processing.dictionary)
embed_dim = 300
save_dir = r"..\save_model"
save_prefix = "TextCNN"

model = TextCNN(vocabulary_size, embed_dim, num_filters, filter_sizes, dropout, num_classes)

training = TrainingModel(model, learning_rate)

print('training...')
training.training_early_stop(inputs_training, target_training, inputs_testing, target_testing, batch_size,
                             num_epochs, save_best=True, save_dir=save_dir, save_prefix=save_prefix)






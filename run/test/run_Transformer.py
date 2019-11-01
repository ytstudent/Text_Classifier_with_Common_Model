import argparse
from model.Transformer_model import Transformer, Config
from training_model import TrainingModel


save_dir = r"..\save_model"
save_prefix = "Transformer"
config = Config()
model = Transformer(config)
early_stop = 5
training = TrainingModel(model, learning_rate=config.learning_rate)
print('training...')
training.training_early_stop(config.inputs_training, config.target_training, config.inputs_testing,
                             config.target_testing,
                             config.batch_size,
                             config.num_epochs, save_best=True, save_dir=save_dir, save_prefix=save_prefix,
                             early_stop=early_stop)

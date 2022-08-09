from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv
from zipfile import ZipFile


def train(model, train_data, val_data, num_epochs=10, batch=16):
     #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    
    #Reading our new data
    train_samples = []
    for idx, row in train_data.iterrows():
            train_samples.append(InputExample(texts=[row['body'], row['body_2']], label=int(row['label'])))
            train_samples.append(InputExample(texts=[row['body_2'], row['body']], label=int(row['label'])))
            


    dev_samples = []
    for idx, row in val_data.iterrows():
            dev_samples.append(InputExample(texts=[row['body'], row['body_2']], label=int(row['label'])))

    #Configuration
    model_save_path = 'output/training_Custome_dataset-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

      
    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch)


    # We add an evaluator, which evaluates the performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='Quora-dev')


    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=5000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)
    
    return model

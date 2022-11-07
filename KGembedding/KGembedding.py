# Deprecated?
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
#from pykeen.datasets.nations import NATIONS_TRAIN_PATH
import os
import pathlib

workDir = pathlib.Path().resolve()
dataDir = os.path.join(workDir.parent.resolve(), 'datasets')
FB15Kdir = os.path.join(dataDir, 'FB15K-237')

training = os.path.join(FB15Kdir, 'train.txt')
testing = os.path.join(FB15Kdir, 'test.txt')
validation = os.path.join(FB15Kdir, 'valid.txt')

result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='TransE',
    dimensions=100,
    epochs=5,  # short epochs for testing - you should go higher
    device='gpu'
)

result.save_to_directory('doctests/test_unstratified_transe')
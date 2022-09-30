from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.datasets.nations import NATIONS_TRAIN_PATH
tf = TriplesFactory.from_path(NATIONS_TRAIN_PATH)
training, testing = tf.split()
result = pipeline(
    training=training,
    testing=testing,
    model='TransE',
    epochs=5,  # short epochs for testing - you should go higher
)
result.save_to_directory('doctests/test_unstratified_transe')




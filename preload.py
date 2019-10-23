from flair.models import SequenceTagger, TextClassifier

from flair_as_service import Configs, InitializedPipeline


raw_pipeline = Configs("./conf/config.json").pipeline

# The following line loads all the models into memory
InitializedPipeline(raw_pipeline)

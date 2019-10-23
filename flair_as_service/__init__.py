import json

from flair.embeddings import (
    BertEmbeddings,
    BytePairEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    FlairEmbeddings,
    StackedEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger, TextClassifier


class Embeder:
    def __init__(self, pipeline):
        self.mode = pipeline.mode
        self.type = pipeline.embedding_type
        embedders = []
        for component in pipeline.embedders:
            if "forward" in component or "backward" in component:
                embedders.append(FlairEmbeddings(component))
            elif "glove" in component:
                embedders.append(WordEmbeddings(component))
            elif "bert" in component:
                embedders.append(BertEmbeddings(component))
            elif len(component) == 2:
                # see https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/FASTTEXT_EMBEDDINGS.md#fasttext-embeddings
                embedders.append(WordEmbeddings(component))
                embedders.append(BytePairEmbeddings(component))
            else:
                raise ValueError(f"unknown embedder: {component}")
        if self.type == "document":
            self.embedder = self._make_doc_embedder(pipeline, embedders)
        elif self.type == "word":
            self.embedder = StackedEmbeddings(embedders)
        elif self.type == "both":
            self.embedders = [
                self._make_doc_embedder(pipeline, embedders),
                StackedEmbeddings(embedders),
            ]
        else:
            raise ValueError(
                f"Innapropriate embedding type {pipeline.embedding_type}, "
                "should be 'word', 'document', or 'both'."
            )

    def _make_doc_embedder(self, pipeline, embedders):
        op = pipeline.pooling_strategy["operation"]
        if op == "pool":
            opts = pipeline.pooling_strategy["pool_options"]
            pooling_type = opts.get("pooling", "mean")
            ft_mode = opts.get("fine_tune_mode", "linear")
            return DocumentPoolEmbeddings(
                embedders, fine_tune_mode=ft_mode, pooling=pooling_type
            )
        elif op == "rnn":
            opts = pipeline.pooling_strategy["rnn_options"]
            hidden_size = opts.get("hidden_size", 128)
            rnn_layers = opts.get("rnn_layers", 1)
            reproject_words = opts.get("reproject_words", True)
            reproject_words_dimension = opts.get("reproject_words_dimension", None)
            bidirectional = opts.get("bidirectional", False)
            dropout = opts.get("dropout", 0.5)
            word_dropout = opts.get("word_dropout", 0.0)
            locked_dropout = opts.get("locked_dropout", 0.0)
            rnn_type = opts.get("rnn_type", "GRU")
            fine_tune = opts.get("fine_tune", True)
            return DocumentRNNEmbeddings(
                embeddings=embedders,
                hidden_size=hidden_size,
                rnn_layers=rnn_layers,
                reproject_words=reproject_words,
                reproject_words_dimension=reproject_words_dimension,
                bidirectional=bidirectional,
                dropout=dropout,
                word_dropout=word_dropout,
                locked_dropout=locked_dropout,
                rnn_type=rnn_type,
                fine_tune=fine_tune,
            )
        else:
            raise ValueError(f"{op} is not a valid pooling strategy operation")

    def _token_embed(self, token):
        if self.mode == "server":
            return token.embedding.cpu().numpy().tolist()
        else:
            return token.embedding

    def _sent_embed(self, sent):
        if self.mode == "server":
            return sent.get_embedding().detach().numpy().tolist()
        else:
            return sent.get_embedding()

    def __call__(self, sentence):
        if self.type == "word":
            self.embedder.embed(sentence)
            return {
                "text": sentence.to_original_text(),
                "tokens": [
                    {"token": t, "embedding": self._token_embed(t)}
                    for t.text in sentence
                ],
            }
        elif self.type == "document":
            self.embedder.embed(sentence)
            return {
                "text": sentence.to_original_text(),
                "embedding": self._sent_embed(sentence),
            }
        elif self.type == "both":
            for embedder in self.embedders:
                embedder.embed(sentence)
            return {
                "text": sentence.to_original_text(),
                "embedding": self._sent_embed(sentence),
                "tokens": [
                    {"token": t.text, "embedding": self._token_embed(t)}
                    for t in sentence
                ],
            }


class Pipeline:
    def __init__(
        self,
        embedding_type=None,
        pooling_strategy=None,
        mode="server",
        taggers=[],
        classifiers=[],
        embedders=[],
    ):
        self.mode = mode
        self.embedding_type = embedding_type
        self.pooling_strategy = pooling_strategy
        self.taggers = taggers
        self.classifiers = classifiers
        self.embedders = embedders


class InitializedPipeline:
    def __init__(self, pipeline: Pipeline):
        if pipeline.taggers:
            taggers = []
            for component in pipeline.taggers:
                taggers.append(SequenceTagger.load(component))
            self.taggers = taggers
        if pipeline.classifiers:
            classifiers = []
            for component in pipeline.classifiers:
                classifiers.append(TextClassifier.load(component))
            self.classifiers = classifiers
        if pipeline.embedders:
            self.embedder = Embeder(pipeline)


class Configs:
    valid_taggers = [
        "ner",
        "ner-ontonotes",
        "chunk",
        "pos",
        "frame",
        "ner-fast",
        "ner-ontonotes-fast",
        "chunk-fast",
        "pos-fast",
        "frame-fast",
        "ner-multi",
        "ner-multi-fast",
        "ner-multi-fast-learn",
        "pos-multi",
        "pos-multi-fast",
        "de-ner",
        "de-ner-germeval",
        "de-pos",
        "de-pos-fine-grained",
        "fr-ner",
        "nl-ner",
    ]

    valid_classifiers = ["en-sentiment", "de-offensive-language"]

    def __init__(self, path_to_configs="../conf/config.json"):
        with open(path_to_configs) as f:
            conf = json.load(f)

        if conf.get("num_cores"):
            self.num_cores = num_cores if num_cores > 1 else 1
        else:
            self.num_cores = 1

        pipeline = conf["pipeline"]
        mode = conf["mode"]
        self.mode = mode

        embedders = pipeline.get("word_embeddings", None)
        embedding_type = pipeline.get("embedding_type", "document")
        pooling_strategy = pipeline.get("pooling_strategy", "pool")

        if embedders:
            self.pipeline = Pipeline(
                mode=mode,
                embedding_type=embedding_type,
                pooling_strategy=pooling_strategy,
                embedders=embedders,
            )
        else:
            self.pipeline = Pipeline(mode=mode)

        taggers = pipeline["taggers"]
        problems = list(filter(lambda x: x not in self.valid_taggers, taggers))
        if problems:
            raise ValueError(
                f"Invalid pipeline taggers component(s): {' '.join(problems)}"
            )
        else:
            self.pipeline.taggers += taggers

        classifiers = pipeline["classifiers"]
        problems = list(filter(lambda x: x not in self.valid_classifiers, classifiers))
        if problems:
            raise ValueError(
                f"Invalid pipeline classifier component(s): {' '.join(problems)}"
            )
        else:
            self.pipeline.classifiers += classifiers

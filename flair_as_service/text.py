import json
from typing import Dict, List

from flair.data import Sentence, segtok_tokenizer
from segtok.segmenter import split_single


class Text:
    def __init__(self, text: str, configs, pipeline):
        self.text = text
        self.sentences = [
            Sentence(sent, use_tokenizer=segtok_tokenizer)
            for sent in split_single(text)
        ]
        self.configs = configs
        self.pipeline = pipeline
        self.results: List[Dict] = []

    def _upsert(
        self,
        l: List[Dict],
        entry: Dict,
        match_key: str = "text",
        merge_key: str = "entities",
    ):
        for d in l:
            if d.get(match_key) and d.get(match_key) == entry[match_key]:
                if d.get(merge_key):
                    d[merge_key] += entry[merge_key]
                else:
                    d[merge_key] = entry[merge_key]
                return l
        l.append(entry)
        return l

    def _is_chunker(self, tagger):
        """
        A hacky check to see if the current tagger is doing syntactic chunking
        """
        return "B-VP" in tagger.tag_dictionary.get_items()

    def tag(self):
        for tagger in self.pipeline.taggers:
            chunking = self._is_chunker(tagger)
            tagger.predict(self.sentences)
            for sentence in self.sentences:
                entry = sentence.to_dict(tag_type=tagger.tag_type)
                if chunking:
                    entry["chunks"] = entry["entities"]
                    del entry["entities"]
                    self.results = self._upsert(self.results, entry, merge_key="chunks")
                else:
                    self.results = self._upsert(self.results, entry)

    def classify(self):
        for classifier in self.pipeline.classifiers:
            classifier.predict(self.sentences)
            for sentence in self.sentences:
                labels = sentence.labels
                sentiment = [[label.value, label.score] for label in labels][0]
                self.results = self._upsert(
                    self.results,
                    {"text": sentence.to_original_text(), "sentiment": sentiment},
                    merge_key="sentiment",
                )

    def embed(self):
        if not getattr(self.pipeline, "embedder", False):
            return
        for sentence in self.sentences:
            res = self.pipeline.embedder(sentence)
            bed_type = self.configs.pipeline.embedding_type
            if bed_type == "word":
                self.results = self._upsert(
                    self.results, res, match_key="text", merge_key="tokens"
                )
            elif bed_type == "document":
                self.results = self._upsert(
                    self.results, res, match_key="text", merge_key="embedding"
                )
            elif bed_type == "both":
                res_word = {"text": res["text"], "tokens": res["tokens"]}
                res_doc = {"text": res["text"], "embedding": res["embedding"]}
                self.results = self._upsert(
                    self.results, res_word, match_key="text", merge_key="tokens"
                )
                self.results = self._upsert(
                    self.results, res_doc, match_key="text", merge_key="embedding"
                )

    def analyze(self):
        self.tag()
        self.classify()
        self.embed()

    def __repr__(self):
        return json.dumps(self.results)

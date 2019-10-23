# flair as service

Run [flair](https://github.com/zalandoresearch/flair) (the NLP library by Zalando research) as a container, configure it with one JSON file, and get a REST API that responds in a compact format (relative to flair's default API shape).

## tl;dr

Try it out:

`docker run -it -p 5000:5000 samhavens/flair-as-service:en-full` then POST or GET http://0.0.0.0:5000/?text=George%20Washington%20went%20to%20the%20store%20and%20choked%20on%20a%20cherry%20pit

## Intro

Create word and/or document embeddings (or span tags, or classifications) that can be used downstream in other services.

The name is intentionally meant to invoke [bert-as-service](https://github.com/hanxiao/bert-as-service), but there are some differences as well as similarities

### Similarities to bert-as-service

* Convert variable length documents to fixed length vectors (word and/or document embeddings)

* Centralize text feature-extraction to one service

* Run on CPU or GPU

### Differences from bert-as-service

* flair-as-service uses flair to get embeddings instead of BERT. This lets you get better performance out of CPU-only mode, but has downsides as well

* flair-as-service expects to be run in a Kubernetes cluster (though it will also work outside of Kubernetes), so rather than using worker processes, you are expected to spin up more containers/pods of fas, and allow k8s to handle the networking (and hence the is no message queue)

* flair-as-service does more than embeddings. It will also do span tagging (such as NER (named entity recognition), semantic chunking (identifying verb/noun-phrases), or custom models) and document classification (sentiment, offensive language, custom models)


## Features

flair-as-service supports embedding, tagging, and classifying.

### Embedding

All valid flair embedding models should work. If you are running flair-as-service in Docker/k8s, keep in mind that there are extra steps to using the GPU, so if you can handle the ~1% accuracy drop, it may be better to stick with CPU models

### Tagging

I have tested some English NER and the POS chunking model. Other valid models should work.

### Classification

Only `en-sentiment` has been tested for now, but `de-offensive-language` should work as well. It shouldn't be hard to add support for custom classification models, though.

## Installation

@todo put on PyPi

The easiest way to run flair-as-service is in Docker/Kubernetes, which is explained below. If you want, you can git clone it and `pip install -r requirements.txt && python application.py` and it should work as well.

## Usage

This section focuses on using the containerized version of flair-as-service, but there is not much logic in the `Dockerfile`, and 

### Quickstart

(If you are a backend or devops engineer trying to help a data scientist, this is what you should start with). If you need to get things running quickly, this one-liner will spin up a REST server which provides NER, semantic POS chunking, sentiment analysis, and document embeddings:

```sh
docker run -it -p 5000:5000 samhavens/flair-as-service:en-full
```

To test it out, send a cURL request or just open your browser to http://0.0.0.0:5000/?text=George%20Washington%20went%20to%20the%20store%20and%20choked%20on%20a%20cherry%20pit

You should see:

```jsonc
[
  {
    "text": "George Washington went to the store and chocked on a cherry pit",
    "labels": [],
    "entities": [
      {
        "text": "George Washington",
        "start_pos": 0,
        "end_pos": 17,
        "type": "PERSON",
        "confidence": 0.9513236582
      }
    ],
    "chunks": [
      {
        "text": "George Washington",
        "start_pos": 0,
        "end_pos": 17,
        "type": "NP",
        "confidence": 0.9786532521
      },
      {
        "text": "went",
        "start_pos": 18,
        "end_pos": 22,
        "type": "VP",
        "confidence": 0.9999723434
      },
      {
        "text": "to",
        "start_pos": 23,
        "end_pos": 25,
        "type": "PP",
        "confidence": 0.9999266863
      },
      {
        "text": "the store",
        "start_pos": 26,
        "end_pos": 35,
        "type": "NP",
        "confidence": 0.9940359592
      },
      {
        "text": "choked",
        "start_pos": 40,
        "end_pos": 47,
        "type": "VP",
        "confidence": 0.9998243451
      },
      {
        "text": "on",
        "start_pos": 48,
        "end_pos": 50,
        "type": "PP",
        "confidence": 0.9908009171
      },
      {
        "text": "a cherry pit",
        "start_pos": 51,
        "end_pos": 63,
        "type": "NP",
        "confidence": 0.9735482136
      }
    ],
    "sentiment": [
      "NEGATIVE",
      0.6297242045
    ],
    "embedding": [
      # (a 2048-dimensional vector that isn't included because it is too long)
    ]
  }
]
```

### Customizing

Different models come preloaded into flair-as-service, depending on Docker tag. For example:

* `en-full` - provides english embeddings, NER, POS chunking, and sentiment
* `en-embedding` - provides english embeddings
* `en-ner` - provides english NER

However, don't go nuts trying to create the tag `de-embeddings-ner-pos` or something. Instead of using Docker's tags for that, use a `config.json` file. See the flair docs for the [list of possible embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md) or the [list of tagging models](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#list-of-pre-trained-sequence-tagger-models).

For example, here is a minimal config for NER and sentiment analysis:

```json
{
    "mode": "server",
    "pipeline": {
        "taggers": [
            "ner-ontonotes-fast"
        ],
        "classifiers": [
            "en-sentiment"
        ],
        "word_embeddings": []
    }
}
```

(Note, if GPU is available, then you want the tagger `ner-ontonotes` (no `-fast`))

`mode` is `server` or `library`, though `library` mode isn't working yet. The idea is that it would be an easier-to-use flair, and hence could keep tensors on the GPU, whereas the server will always need to move them to CPU and convert them to lists.

To run flair-as-service with a custom config file, use a volume:

```sh
docker run -it -p 5000:5000 -v path/to/custom_config.json:/app/conf/config.json samhavens/flair-as-service:en-ner
```

### Model downloads

If you build your own Docker image, it will preload all models you need into the image you built. If you change the config before running, then at startup time, the model will download before running.

## Todo

The most immediate todo is to make batching the default behavior and to write a client. Also, put on PyPi so it is pip installable.

# Contact

Probably the quickest way to get my attention is to bug [me on Twitter](https://twitter.com/sam_havens)

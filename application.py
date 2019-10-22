from flair_as_service.text import Text
from flair_as_service import Configs, InitializedPipeline

configs = Configs("./conf/config.json")

if configs.mode == "server":
    from sanic import Sanic
    from sanic.response import json

raw_pipeline = configs.pipeline
pipeline = InitializedPipeline(raw_pipeline)


def make_text(sent):
    return Text(sent, configs=configs, pipeline=pipeline)


if __name__ == "__main__":

    app = Sanic()

    @app.route("/")
    async def process(request):
        try:
            in_text = request.json["text"]
        except:
            in_text = request.args["text"][0]
        text = make_text(in_text)
        text.analyze()
        return json(text.results)

    app.run(host="0.0.0.0", port=5000)


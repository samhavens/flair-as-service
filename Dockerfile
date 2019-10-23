FROM python:3.7

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && pip install -r requirements.txt

# https://github.com/py-bson/bson/issues/82
# flair requires pymongo, but there's some messed up monkeypatching going on

RUN pip uninstall -y bson pymongo \
    && pip install bson==0.5.7 \
    && pip install pymongo==3.7.2

RUN python preload.py

CMD ["python", "application.py"]

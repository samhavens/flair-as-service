FROM python:3.7

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && pip install -r requirements.txt \
    && python preload.py

CMD ["python", "application.py"]

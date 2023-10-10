FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    build-essential \
    cifs-utils \
    curl \
    default-jdk \
    dialog \
    dos2unix \
    git \
    sudo

# Install app requirements first to avoid invalidating the cache
COPY requirements.txt setup.py /app/
WORKDIR /app
RUN pip install --user -r requirements.txt --no-warn-script-location && \
    pip install --user entmax && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

RUN pip install -U typing_extensions && \
    pip install stanfordnlp

# Cache the pretrained BERT model
# RUN python -c "from transformers import BertModel; BertModel.from_pretrained('bert-large-uncased-whole-word-masking')"

# Download & cache StanfordNLP
RUN mkdir -p /app/third_party && \
    cd /app/third_party && \
    curl https://download.cs.stanford.edu/nlp/software/stanford-corenlp-full-2018-10-05.zip | jar xv


# Assume that the rat-sql project will be mounted as a volume into /mnt on startup.
# Symlink the data subdirectory to that volume.
ENV CACHE_DIR=/mnt
RUN ln -snf /mnt /app/mnt

# Convert all shell scripts to Unix line endings, if any
RUN /bin/bash -c 'if compgen -G "/app/**/*.sh" > /dev/null; then dos2unix /app/**/*.sh; fi'

# Extend PYTHONPATH to load WikiSQL dependencies
ENV PYTHONPATH="/app/third_party/wikisql/:${PYTHONPATH}" 

RUN pip install sqlparse
# Additional for wtq evaluator
RUN pip install stanfordnlp
RUN apt-get update -y \
    && apt-get install -y nodejs npm

ENTRYPOINT bash

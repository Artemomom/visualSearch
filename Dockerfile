FROM gcr.io/gd-gcp-rnd-concept-search/image-search-base-fashion:0.3.0

ENV GCSFUSE_REPO gcsfuse-buster

# Install gcsfuse
RUN apt-get update && apt-get install --yes --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install --yes gcsfuse \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

WORKDIR /usr/src/app/
COPY ./modules /usr/src/app/modules/
COPY ./server.py /usr/src/app/
COPY ./gunicorn.py /usr/src/app/

RUN mkdir -p /usr/src/app/index /usr/src/app/model

ENV NUM_WORKERS=1
ENV PYTHONUNBUFFERED=1

ARG STYLE_COEFFICIENT=0
ENV STYLE_COEFFICIENT=STYLE_COEFFICIENT

CMD ["gunicorn", "--timeout", "180", "-c", "gunicorn.py", "server:app"]

EXPOSE 5000

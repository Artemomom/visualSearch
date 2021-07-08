FROM gcr.io/gd-gcp-rnd-concept-search/image-search-base-fashion:0.3.0

WORKDIR /usr/src/app/

COPY ./modules /usr/src/app/modules/
COPY ./server.py /usr/src/app/
COPY ./gunicorn.py /usr/src/app/

VOLUME /usr/src/app/index
VOLUME /usr/src/app/model

ENV NUM_WORKERS=2
ENV PYTHONUNBUFFERED=1

ARG STYLE_COEFFICIENT=0
ARG CONTENT_COEFFICIENT=1
ENV STYLE_COEFFICIENT=$STYLE_COEFFICIENT
ENV CONTENT_COEFFICIENT=$CONTENT_COEFFICIENT


CMD ["gunicorn", "--timeout", "180", "-c", "gunicorn.py", "server:app"]

EXPOSE 5000

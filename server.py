import io
import logging
import os
import pickle
from base64 import b64encode
from io import BytesIO
from typing import List

import requests
from PIL import Image
from flask import Flask, request, jsonify, abort
import numpy as np

from modules.embedder import Embedder
from modules.indexer import KNN
from modules.searcher import SimilaritySearcher
from modules.timer import Timer

import cv2

style_coef = float(os.environ['STYLE_COEFFICIENT'])
content_coef = float(os.environ['CONTENT_COEFFICIENT'])

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

n_images = 25
style_embedding = "model/imgs2style_pca.pickle"
content_embedding = "model/imgs2embs_res.pickle"
pca_model= "model/pca_model.pickle"


embeder = Embedder(pca_model)
index = KNN(128, style_embedding,content_embedding, style_coef,content_coef)

similarity_searcher = SimilaritySearcher(embeder, index)


def ids2path(ids) -> List[List[str]]:
    result_list = []
    for item in ids:
        image_url = f"https://storage.googleapis.com/street2shop_public_images/floralImages/{item}"
        product_id = str(item)
        result_list.append([image_url, product_id, "Image Name"])
    return result_list


def convert_img_to_base64(image, max_size):
    file_object = io.BytesIO()
    image = Image.fromarray(image.astype('uint8'))
    image.thumbnail((max_size, max_size))
    image.save(file_object, 'PNG')
    return "data:image/png;base64,{}".format(b64encode(file_object.getvalue()).decode('ascii'))

def search_similar(timer: Timer):
    if "image" in request.files and request.files["image"].filename != "":
        np_file = np.fromfile(request.files["image"], np.uint8)
        src_image = cv2.imdecode(np_file, cv2.IMREAD_COLOR)
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    else:
        url = request.form["url"]
        r = requests.get(url, allow_redirects=True)
        stream = BytesIO(r.content)
        image = Image.open(stream).convert("RGB")
        stream.close()
        src_image = np.array(image)
    target_image = src_image
    dists, ids = similarity_searcher.search_image(target_image, n_images=n_images, timer=timer)
    similar_paths = ids2path(ids)

    src_image = convert_img_to_base64(src_image, 1200)
    target_image = convert_img_to_base64(target_image, 400)

    return target_image, similar_paths, src_image


@app.route('/process', methods=['POST'])
def process():
    if request.files or request.form:
        if "image" in request.files and request.files["image"].filename == "" and request.form["url"] == "":
            return jsonify({})
        timer = Timer()
        with timer.measure('total'):
            img_after_segmentation, similar_paths, src_image = search_similar(timer)

        time_stats = {k.replace(" ", "_"): f'{v:.2f}' for k, v in timer.measurements.items()}
        app.logger.info(time_stats)
        model_version = f"VGG19 and Resnext50 pretrained models with style: {style_coef} and content: {content_coef}" \
                        f" coefficients"

        stats = {'index_size': len(similarity_searcher.labels), 'perf': time_stats,
                 'model_version': model_version}
        response = jsonify({"similar_paths": similar_paths, "img_after_segmentation": img_after_segmentation,
                            "target_img": src_image, "stats": stats})

        add_server_timing_header(response, timer.measurements)
        return response
    return abort(400)


def add_server_timing_header(response, time_stats):
    """Adds the Server-Timing header to the response, in accordance with: https://w3c.github.io/server-timing/
    """
    timing_list = [f'{k.replace(" ", "_")};dur={v * 1000:.2f}' for k, v in time_stats.items()]
    response.headers.set('Server-Timing', ', '.join(timing_list))
    return response


if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')

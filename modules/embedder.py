from torchvision.transforms.functional import to_pil_image

from modules.augs import prepare_data_transforms
from modules.model import load_style_model,load_resnext_model
from modules.model import StyleModel
from sklearn.preprocessing import normalize
import pickle

import numpy as np
import torch




class Embedder(object):
    def __init__(self,pca_path) -> None:
        with open(pca_path, "rb") as f:
            self.pca_model = pickle.load(f)
        models = [load_style_model(), load_resnext_model()]
        for m in models:
            m.to("cpu")
            m.eval()
        self.models = models
        self.transforms = prepare_data_transforms(224, 'val')

    def embed_image(self, img:np.ndarray):
        vectors = []
        with torch.no_grad():
            input = self.transforms(to_pil_image(img))
            for m in self.models:
                vec = m(input.unsqueeze(0)).cpu()
                if isinstance(m, StyleModel):
                    vec = torch.flatten(vec).unsqueeze(0)
                    vec = torch.tensor(self.pca_model.transform(normalize(vec)))
                vec.float().numpy()
                vectors.append(vec)
        return vectors

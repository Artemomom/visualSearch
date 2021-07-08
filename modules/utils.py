import os
from PIL import Image
from pytorch_metric_learning import testers
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from modules.augs import prepare_data_transforms
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import glob
import torch
import argparse
import pickle
from modules.model import load_style_model, load_resnext_model


class SimpleImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, images, transform=None):
        self.images = images
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_filename = self.images[idx]
            image = Image.open(os.path.join(self.img_folder, image_filename)).convert('RGB')
            code = 0
        except:
            image = Image.new(mode="RGB", size=(100, 100))
            code = -1
        if self.transform:
            image = self.transform(image)
        return image, code


def load_image(img_path, device, max_size=400, shape=None):
    image = Image.open(img_path)

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    prepare_image = prepare_data_transforms(size, "val")

    image = prepare_image(image).unsqueeze(0)

    return image.to(device, torch.float)


def style_to_vec(style):
    return torch.flatten(style).float().numpy()


def produce_embeddings(folder_path, path_to_save,path_to_pca, style=False):
    style_imgs = glob.glob(folder_path + "/*")

    images = list(map(os.path.basename, style_imgs))

    dataset = SimpleImageFolderDataset(folder_path, images, prepare_data_transforms(224, "val"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tester = testers.BaseTester(batch_size=128, data_device=device, normalize_embeddings=True)

    if style:
        model = load_style_model()
        pca = PCA(n_components=2048)
        image_style_embeddings = {}
        for image_path in tqdm(style_imgs):
            image_tensor = load_image(image_path, device)

            style = style_to_vec(model(image_tensor))
            image_style_embeddings[os.path.basename(image_path)] = style
        i, v = zip(*image_style_embeddings.items())
        vectors = np.asarray(v)
        normalized = normalize(vectors)
        pca_model = pca.fit(normalized)
        embeddings_pca = pca_model.transform(normalized)
        img2emb_style = dict(zip(i, embeddings_pca))
        with open(path_to_save, "wb") as f:
            pickle.dump(img2emb_style, f)

        with open(path_to_pca, "wb") as f:
            pickle.dump(pca_model, f)
    else:
        model = load_resnext_model()
        with autocast(True):
            embeddings,codes = tester.get_all_embeddings(dataset,model)
        img2emb_res = dict(zip(images, embeddings))
        with open(path_to_save, "wb") as f:
            pickle.dump(img2emb_res, f)


def main():
    argparser = argparse.ArgumentParser(description="specify style or content embeddings and path were to store")
    argparser.add_argument('-p', '--images_path', required=False,
                           default="/Users/aivashchenko/Documents/floralImages",
                           help='to specify path where images are stored')

    argparser.add_argument('-pca', '--pca_path', required=False,
                           default="/Users/aivashchenko/Documents/floralImages/pca_model.pickle",
                           help='to specify path where pca model to store')

    argparser.add_argument('-ps', '--store_path', required=False,
                           default="/Users/aivashchenko/Documents/embeddings.pickle",
                           help='to specify path where images are stored')

    argparser.add_argument('-s', '--style', required=False, action='store_true',
                           default=True)

    args = argparser.parse_args()

    produce_embeddings(args.images_path, args.store_path, args.style)


if __name__ == "__main__":
    exit(main())

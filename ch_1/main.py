from __future__ import annotations
import os
import faiss
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import json
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import supervision as sv



transform_image = T.Compose([
		T.ToTensor(),
		T.Resize(224),
		T.CenterCrop(224),
		T.Normalize([0.5], [0.5])])

def load_image(img: str) -> torch.Tensor:
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img


def create_index(files: list, model) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(384)
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = model(load_image(file).to(device))
            embedding = embeddings[0].cpu().numpy()
            all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()
            index.add(np.array(embedding).reshape(1, -1))

    with open("all_embeddings.json", "w") as f:
	    f.write(json.dumps(all_embeddings))

    faiss.write_index(index, "data.bin")

    return index, all_embeddings



def search_index(index: faiss.IndexFlatL2, embeddings: list, k: int = 3) -> list:
    D, I = index.search(np.array(embeddings[0].reshape(1, -1)), k)

    return I[0]


if __name__ == "__main__":
    cwd = os.getcwd()

    ROOT_DIR = os.path.join(cwd, "COCO-128-2/train/")

    files = os.listdir(ROOT_DIR)
    files = [os.path.join(ROOT_DIR, f) for f in files if f.lower().endswith(".jpg")]

    ###

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)
    data_index, all_embeddings = create_index(files, dinov2_vits14)

    search_file = "COCO-128-2/valid/000000000081_jpg.rf.5262c2db56ea4568d7d32def1bde3d06.jpg"
    img = cv2.resize(cv2.imread(search_file), (416, 416))

    with torch.no_grad():
        embedding = dinov2_vits14(load_image(search_file).to(device))
        indices = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1))

        for i, index in enumerate(indices):
            print(f"Image {i}: {files[index]}")
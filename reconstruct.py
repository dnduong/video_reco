import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
import cv2
import logging
import coloredlogs

coloredlogs.install(level="info", fmt="%(asctime)s %(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser(description="Reconstruction")
    parser.add_argument("--input_video_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--number_best_videos", type=int, default=5, help="Output top n videos")

    return vars(parser.parse_args())


def video_to_frames(video_path):
    logger.info("Extracting frames from video")
    frames = []
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, frame = video.read()
        try:
            if success:
                frames.append(frame)
                count = count + 1
        except Exception as e:
            logger.error(e)
    logger.info("{} extracted".format(len(frames)))
    return frames


def resnet34_feature_extractor():
    # Define resnet34 model and pre-processing transforms
    model = resnet34(weights="DEFAULT")
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def extract_features(image, model, transform):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor.numpy().flatten()


def cosine_similarity_matrix(features):
    # Compute similarity matrix
    n_images = len(features)
    similarity_matrix = np.zeros((n_images, n_images))
    for i in range(n_images):
        for j in range(n_images):
            if i == j:
                similarity_matrix[i, j] = -1
            elif i < j:
                similarity_matrix[i, j] = cosine_similarity(features[i].reshape(1, -1), features[j].reshape(1, -1))[0][
                    0
                ]
            else:
                similarity_matrix[i, j] = similarity_matrix[j, i]
    return similarity_matrix


def write_video(frames, fps, output_path, frame_size=(1920, 1080)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def get_top_n_videos(similarity_matrix, frames, n):
    # Sort frames by similarity scores
    videos = []
    avg_similarites = {}
    n_images = similarity_matrix.shape[0]
    for curr_idx in range(n_images):
        similarity_matrix_copy = similarity_matrix.copy()
        sorted_idx = []
        sim = []
        saved_idx = curr_idx
        sorted_idx.append(curr_idx)
        for i in range(n_images):
            # Find the index of the most similar frame that has not been used yet
            for idx in sorted_idx:
                similarity_matrix_copy[curr_idx][idx] = -1
            max_index = np.argmax(similarity_matrix_copy[curr_idx])
            max_similarity = similarity_matrix_copy[curr_idx][max_index]
            if max_similarity > 0.9:
                sorted_idx.append(max_index)
                sim.append(max_similarity)
            curr_idx = max_index
        if len(sim) > 0:
            videos.append(sorted_idx)
            avg_similarites[saved_idx] = np.mean(sim)

    avg_similarites = {k: v for k, v in sorted(avg_similarites.items(), key=lambda item: item[1], reverse=True)[:n]}
    corrected_videos = [[frames[i] for i in videos[best_idx]] for best_idx in avg_similarites.keys()]
    return corrected_videos


if __name__ == "__main__":
    args = arg_parse()

    os.makedirs(args["output_folder"], exist_ok=True)

    frames = video_to_frames(args["input_video_path"])

    feature_extractor, transform = resnet34_feature_extractor()

    logger.info("Extracting features from frames")
    # Extract features for all frames
    features = [extract_features(img, feature_extractor, transform) for img in frames]

    logger.info("Getting similarity matrix")
    similarity_matrix = cosine_similarity_matrix(features)

    top_n_videos = get_top_n_videos(similarity_matrix, frames, args["number_best_videos"])

    for i, frames in enumerate(top_n_videos):
        output_path = os.path.join(args["output_folder"], "top_{}.mp4".format(i))
        logger.info("Save video to {}".format(output_path))
        write_video(frames, args["fps"], output_path)

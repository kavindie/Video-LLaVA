
import torch
from dpp_utils import kernel_simple
from tune_embeddings import qvhighlights_topk_samples
import re
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import imageio
from moviepy import *

def plot_videos_from_images(img_collections, output_file="combined_videos.mp4", fps=15):
    # Create the combined video frames
    num_videos = len(img_collections)
    max_frames = max(len(img_collections[i]) for i in range(num_videos))
    combined_frames = []
    for frame_idx in range(max_frames):
        fig = plt.figure(figsize=(num_videos * 5, 5)) # Adjust figure size as needed.
        gs = gridspec.GridSpec(1, num_videos)

        for video_idx, video_frames in enumerate(img_collections):
            ax = fig.add_subplot(gs[0, video_idx])
            ax.axis('off')

            if frame_idx < len(video_frames):
                ax.imshow(video_frames[frame_idx])
            else:
                # If a video has fewer frames, display a blank image.
                ax.imshow(np.zeros_like(img_collections[0][0]))

        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        img_array_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR) # convert to BGR for cv2
        combined_frames.append(img_array_cv2)
        plt.close(fig) # close figure to prevent memory issues.

    # Write the combined frames to a video file
    height, width, _ = combined_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in combined_frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Combined video saved to {output_file}")

def get_relevant_vid_frames(image_folder, sorted_image_files, index, SEGMENT_LENGTH=8):
    start_index = index*SEGMENT_LENGTH
    end_index = start_index + SEGMENT_LENGTH
    relevant_frames = [np.array(Image.open(os.path.join(image_folder, sorted_image_files[i]))) for i in range(start_index, end_index)]
    return relevant_frames

def create_gif_from_figs(images, output_filename="output", fps=10):
    # images = []
    # for fig in figs:
    #     fig.canvas.draw()  # Draw the figure onto the canvas
    #     img_array = np.array(fig.canvas.renderer.buffer_rgba())
    #     images.append(img_array)
    #     plt.close(fig) # close figure to prevent memory issues.
    
    # gif
    imageio.mimsave(f'{output_filename}.gif', images, fps=fps)
    print(f"GIF saved")

    # video
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(f'{output_filename}.mp4', fps=fps)
    print(f"video saved")

def plot_3d_pose(ax, df, index,  SEGMENT_LENGTH=8, color='b', alpha=0.5, only2D=True):
    if index is None:
        start_index = 0
        end_index = len(df)
    else:
        start_index = index*SEGMENT_LENGTH
        end_index = start_index + SEGMENT_LENGTH

    df_range = df.iloc[start_index:end_index]

    x = df_range['x']
    y = df_range['y']
    z = df_range['z']

    
    if only2D:
        # create 2D plot
        ax.plot(x, y, color=color, alpha=alpha)
    else:
        # Create 3D plot
        ax.plot(x, y, z, color=color, alpha=alpha)

def main():
    SEGMENT_LENGTH = 8
    OVERLAP = 0
    IMAGE_FOLDER_vISION = '/scratch3/kat049/datasets/WildScenes/WildScenes2D/v-01/data/image'
    IMAGE_FOLDER_QUERY = '/scratch3/kat049/datasets/WildScenes/WildScenes2D/v-02/data/image'
    DEVICE = "cuda:2"
    TRAIN = False
    
    vision_embeddings =  torch.load(f'/scratch3/kat049/datasets/WildScenes/WildScenes2D/v-01/seg{SEGMENT_LENGTH}_overlap{OVERLAP}/segment_embeddings.pt')
    query_embeddings =  torch.load(f'/scratch3/kat049/datasets/WildScenes/WildScenes2D/v-02/seg{SEGMENT_LENGTH}_overlap{OVERLAP}/segment_embeddings.pt')
    
    figs = []
    images = []
    for query_index in range(2, query_embeddings.shape[0]):
        query_embedding = query_embeddings[query_index]

        L = kernel_simple(vision_embeddings, query_embedding)
        # best_samples = qvhighlights_topk_samples(L, do_L_norm=False)
        # best_sample = L.diag().argmax().item()

        similarity = vision_embeddings @ query_embedding
        sim_idx = similarity.argmax().item()

        df_queries = pd.read_csv('/scratch3/kat049/datasets/WildScenes/WildScenes2D/v-02/metadata/poses2d.csv', sep=' ')
        df_vision = pd.read_csv('/scratch3/kat049/datasets/WildScenes/WildScenes2D/v-01/metadata/poses2d.csv', sep=' ')

        fig = plt.figure()
        plot2D = True
        if plot2D:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        # if you want to plot all the data
        plot_3d_pose(ax, df_queries, None,  SEGMENT_LENGTH, 'b', 0.1, plot2D)
        plot_3d_pose(ax, df_vision, None,  SEGMENT_LENGTH, 'r', 0.1, plot2D)

        plot_3d_pose(ax, df_queries, query_index,  SEGMENT_LENGTH, 'b', 0.5, plot2D)
        # for i in best_samples:
        #     plot_3d_pose(ax, df_vision, i,  SEGMENT_LENGTH, 'r', 0.5, plot2D)
        # plot_3d_pose(ax, df_vision, best_sample,  SEGMENT_LENGTH, 'g', 0.5, plot2D)
        plot_3d_pose(ax, df_vision, sim_idx,  SEGMENT_LENGTH, 'y', 0.5, plot2D)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if not plot2D:
            ax.set_zlabel('Z')
        plt.legend(['Queried path', 'Alt path', 'Queries seg', 'Alt seg sim']) 
        figs.append(fig)
        
        fig.canvas.draw()  # Draw the figure onto the canvas
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        images.append(img_array)
        plt.savefig('text.png')
        plt.close(fig)
    
    create_gif_from_figs(images)

    # if you want to plot videos
    # image_files = [f for f in os.listdir(IMAGE_FOLDER_QUERY) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # sorted_image_files_query = sorted(image_files, key=lambda x: int(re.match(r'(\d+)-', x).group(1)) if re.match(r'(\d+)-', x) else 0)
    # query_frames = get_relevant_vid_frames(IMAGE_FOLDER_QUERY, sorted_image_files_query, query_index, SEGMENT_LENGTH)
    
    # image_files = [f for f in os.listdir(IMAGE_FOLDER_vISION) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # sorted_image_files_vision = sorted(image_files, key=lambda x: int(re.match(r'(\d+)-', x).group(1)) if re.match(r'(\d+)-', x) else 0)
    # matched_framess = []
    # for i in best_samples:
    #     matched_framess.append(get_relevant_vid_frames(IMAGE_FOLDER_vISION, sorted_image_files_vision, i, SEGMENT_LENGTH))

    # plot_videos_from_images([query_frames, *matched_framess])

if __name__ == '__main__':
    main()
import textwrap
import torch
import os
from collections import Counter, defaultdict
import tqdm
from moviepy import *
from PIL import Image
from scipy.optimize import minimize  # For optimization
from objective_functions import compare_embeds, solve_with_lasso, DPP
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from pathlib import Path
from matplotlib import pyplot as plt
import math
import torch.optim as optim
from torch.nn.parallel import parallel_apply
from dppy.finite_dpps import FiniteDPP
import gradio as gr
import time
import numpy as np
import functools
from dpp_utils import *

import os
import pickle
import pandas as pd
import re


SEGMENT_LENGTH = 8  # Segement length in frames
OVERLAP = 1  # Overlap between segments in frames
# --- Hyperparameters (Adjust these) ---
VIDEO_FOLDER = "/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/video" # Path to the folder containing videos
VIDEO_READING_FREQUENCY = 1  # Read one frame every `VIDEO_READING_FREQUENCY` seconds
TRAIN = True
device = "cuda:0"

def plot_selections(video, video_reading_frequency, samples, path='text.png'):
    if video_reading_frequency is None:
        Warning("Video reading frequency is not provided. Defaulting to 1.")
        video_reading_frequency = 1

    num_cols = math.ceil(len(samples) / 4)
    plt.close()
    _, axes = plt.subplots(4, num_cols, tight_layout=True)
    axes = axes.flat
    seq = 0
    for i, frame in enumerate(video.iter_frames(fps=video_reading_frequency, dtype="uint8")):
        if i in samples:
            axes[seq].imshow(frame)
            seq += 1
        if seq > len(samples):
            break
    for ax in axes:
        ax.axis('off')
    plt.savefig(path)
    plt.close()


def get_user_feedback_annotated(frame_mean_values, samples):
    u = frame_mean_values.take(samples)
    median = u.median()
    if median > 3:
        return 1.0 # thumps up
    else:
        return -1.0 # thumps down


class ClusterManager:
    def __init__(self, threshold=0.3):  # Threshold is weird
        self.clusters = []  # List of cluster centers (tensors)
        self.threshold = threshold

    def add_tensor(self, new_tensor):
        """Adds a new tensor to the appropriate cluster or starts a new one."""

        if not self.clusters:  # No clusters yet, start the first one
            self.clusters.append(new_tensor)
            return

        best_cluster_index = -1
        max_similarity = -1

        for i, cluster_center in enumerate(self.clusters):
            similarity = self.calculate_similarity(new_tensor, cluster_center)  # Define similarity function
            if similarity > max_similarity:
                max_similarity = similarity
                best_cluster_index = i

        if max_similarity >= self.threshold:  # Add to existing cluster
            updated_cluster_center = self.update_cluster_center(new_tensor, self.clusters[best_cluster_index])
            self.clusters[best_cluster_index] = updated_cluster_center  # Update the cluster center
        else:  # Start a new cluster
            self.clusters.append(new_tensor)

    def calculate_similarity(self, tensor1, tensor2):
        """Calculates the similarity between two tensors (e.g., cosine similarity)."""
        # Example: Cosine similarity
        # Calculate L2 norm along the last dimension (embedding dimension)
        normalized_tensor_1 = tensor1 / torch.norm(tensor1, p=2, dim=-1, keepdim=True)
        normalized_tensor_2 = tensor2 / torch.norm(tensor2, p=2, dim=-1, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(normalized_tensor_1[0], normalized_tensor_2[0], dim=0)  # Similarity along the 257 dimension
        similarity = similarity.abs().mean()
        return similarity.item() # Return the similarity value as a float.

    def update_cluster_center(self, new_tensor, current_center):
        """Updates the cluster center (e.g., by averaging)."""
        # Example: Averaging
        new_center = (new_tensor + current_center) / 2
        return new_center


class UserAdaptation(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super(UserAdaptation, self).__init__()
        self.device = device
        self.linear = torch.nn.Linear(input_size, output_size).to(self.device)
        if output_size == input_size:
            torch.nn.init.eye_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)
    def forward(self, x):
        # W_sym = (self.linear.weight + self.linear.weight.t()) / 2
        # y = torch.matmul(x, W_sym.t()) + self.linear.bias # making sure this is symmetric
        y = self.linear (x)
        return y
    
class Incremental_VQA_LanguageBind():
    def __init__(self, segment_size=8, overlap=1, device='cpu', train=True, trained_output_size=None):
        self.device = device
        self.train = train
        clip_type = {
            'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
            'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
            'thermal': 'LanguageBind_Thermal',
            'image': 'LanguageBind_Image',
            'depth': 'LanguageBind_Depth',
        }
        self.LanguageBindModel = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        self.LanguageBindModel = self.LanguageBindModel.to(self.device)
        self.LanguageBindModel.eval()
        pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
        self.modality_transform = {c: transform_dict[c](self.LanguageBindModel.modality_config[c]) for c in clip_type.keys()}
        print("LanguageBind loaded")


        if segment_size > 8:
            raise ValueError("Segment size should be less than or equal to 8")
        self.segment_size = segment_size

        if overlap > segment_size or overlap < 0:
            raise ValueError("Overlap should be between 0 and segment_size")
        self.overlap = overlap

        self.frames = []

        self.frame_embeddings = []
        self.segment_embeddings = []

        # self.cluster_manager_frames = ClusterManager()
        # self.cluster_manager_segments = ClusterManager()

        self.dummy_text = [""]
        self.lambda_param = 0.01

        if trained_output_size is None:
            trained_output_size = 768
        self.learn_user_preferences = UserAdaptation(input_size=768, output_size=trained_output_size, device=self.device)

    def reset(self):
        self.frames = []
        self.frame_embeddings = []
        self.segment_embeddings = []

    def process_new_frame(self, frame):
        """frame: np.ndarray"""
        inputs = {}
        inputs = {
            'image': to_device(self.modality_transform['image'](frame), self.device),
        }        

        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)['image'][0] # todo need to save this  
            ## clustering 
            ## full_embeddings = self.LanguageBindModel.modality_encoder['image'](inputs['image']['pixel_values'])[0]
            ## self.cluster_manager_frames.add_tensor(full_embeddings)

        if self.train:
            embeddings = self.learn_user_preferences(embeddings)
        else:
            with torch.no_grad():
                embeddings = self.learn_user_preferences(embeddings)

        self.frame_embeddings.append(embeddings)
        self.frames.append(frame)

        if len(self.frames) == self.segment_size:
            self.process_segment()
            self.frames = [] if self.overlap == 0 else self.frames[-self.overlap:]

    def process_segment(self):
        # frames_tensor = torch.stack(self.frames).to(self.device)  # Shape: (seq_len, height, width, 3)
        frames_tensor = torch.tensor(np.array(self.frames)).to(self.device)  # Shape: (seq_len, height, width, 3)
        # print("frames_tensor.shape", frames_tensor.shape)
        inputs= {}

        transform = self.modality_transform['video'].transform
        video_data = frames_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)
        video_outputs_dict = {"pixel_values": video_outputs.unsqueeze(0)}

        inputs = {
            'video': to_device(video_outputs_dict, self.device),
        }                
        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)['video'][0] # todo need to save this

            ## clustering 
            ## full_embeddings = self.LanguageBindModel.modality_encoder['video'](inputs['video']['pixel_values'])[0]
            ## self.cluster_manager_segments.add_tensor(full_embeddings)

        if self.train:
            embeddings = self.learn_user_preferences(embeddings)
        else:
            with torch.no_grad():
                embeddings = self.learn_user_preferences(embeddings)
        
            
        self.segment_embeddings.append(embeddings)

        # if len(self.frames) == self.segment_size:
        #     self.frames = [] if self.overlap == 0 else self.frames[-self.overlap:]

    # def loss(self, user_mask, mask):
    #     FP = ((user_mask & ~mask).float()).mean()
    #     TP = ((user_mask & mask).float()).mean()
    #     FN = ((~user_mask & mask).float()).mean()
    #     TN = ((~user_mask & ~mask).float()).mean()
    
    def get_question_embedding(self, question):
        language = question
        inputs = {}     
        inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)
        
        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)['language'][0]
        
        if self.train:
            embeddings = self.learn_user_preferences(embeddings)
        else:
            with torch.no_grad():
                embeddings = self.learn_user_preferences(embeddings)
        return embeddings

    def answer_question(self, question, video = None, video_reading_frequency = 1, output='images'):
        # inputs can have , output_format='images', user_mask_frames=None, user_mask_segments=None
        # self.process_segment()
        language = question
        inputs = {}     
        inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)
        
        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)['language'][0]
        
        if self.train:
            embeddings = self.learn_user_preferences(embeddings)
        else:
            with torch.no_grad():
                embeddings = self.learn_user_preferences(embeddings)
        
        question_embedding = embeddings
        if output == 'images':
            vision_embeddings = torch.stack(self.frame_embeddings)
        else:
            vision_embeddings = torch.stack(self.segment_embeddings)
        
        # pdf, best_x = compare_embeds(question_embedding.detach(), vision_embeddings.detach())
        # pdf.argmax().item()
        # mask = torch.tensor(best_x, dtype=torch.bool)  

        
        # B = vision_embeddings*question_embedding
        # B = vision_embeddings * (0.5+ 0.5*(vision_embeddings @ question_embedding) / (vision_embeddings.norm(dim=-1) * question_embedding.norm(dim=-1)) )[:,None]
        # L = B @ B.T
        L = kernel_simple(vision_embeddings, question_embedding)
        L_norm = L / L.diag().max()        
        dpp = FiniteDPP("likelihood", L = L_norm.detach().cpu())
        print('expected_num_frames =', expected_num_frames(L = L_norm))
        while True:
            try:
                samples = dpp.sample_exact()
                break
            except ValueError as e:
                print(e)
            print("...trying again...")
        print('actual   num frames =', len(samples))
        samples.sort()
        if video is not None:
            if output == 'images':
                plot_selections(video, VIDEO_READING_FREQUENCY, samples=samples, path='text.png')

        return samples, L
    

        max_prob = 0
        selected_samples = None
        for k in range(20):
            samples = dpp.sample_exact()
            prob = self.dpp_probability(B, samples)
            if max_prob < prob:
                max_prob = prob
                selected_samples = samples
        selected_samples.sort()


        if video is not None:
            plot_selections(video, video_reading_frequency, samples=selected_samples, path='text.png')
        
        return selected_samples, B
        # segment_embeddings = torch.stack(self.segment_embeddings).detach()
        # if output_format == 'video':
        #     pass
        #     # TODO combine the part of the video
        # elif output_format == 'images':
        #     pdf, best_x = compare_embeds(question_embedding, frame_embeddings) 



def save_emebeddings(embeddings, embedding_saving_path):
    embeddings = torch.stack(embeddings).detach()
    torch.save(embeddings, embedding_saving_path)

def define_video_file_numbers():
    order = ['AwmHb44_ouw', '98MoyGZKHXc', 'J0nA4VgnoCo', 'gzDbaEs1Rlg', 'XzYM3PfTM4w', 'HT5vyqe0Xaw', 'sTEELN-vY30', 'vdmoEJ5YbrQ', 'xwqBXPGE9pQ', 'akI8YFjEmUw', 'i3wAGJaaktw', 'Bhxk-O1Y7Ho', '0tmA_C6XwfM', '3eYKfiOEJNs', 'xxdtq8mxegs', 'WG0MBPpPC6I', 'Hl-__g2gn_A', 'Yi4Ij2NM7U4', '37rzWOQsNIw', 'LRw_obCPUt0', 'cjibtmSLxQ4', 'b626MiF1ew4', 'XkqCExn6_Us', 'GsAD1KT1xo8', 'PJrm840pAUI', '91IHQYk1IQM', 'RBCABdttQmI', 'z_6gVvQb2d0', 'fWutDQy1nnY', '4wU_LUjG5Ic', 'VuWGsYPqAX8', 'JKpqYvAdIsw', 'xmEERLqJ2kU', 'byxOvuiIJV0', '_xMr-HKMfVA', 'WxtbjNsCQ8A', 'uGu_10sucQo', 'EE-bNr36nyA', 'Se3oxnaPsz0', 'oDXZc0tZe04', 'qqR6AEXwxoQ', 'EYqVtI9YWJA', 'eQu1rNs0an0', 'JgHubY5Vw3Y', 'iVt07TCkFM0', 'E11zDS9XGzg', 'NyBmCxDoHJU', 'kLxoNp-UchI', 'jcoYJXDG9sw', '-esJrBWj2d8', ]
    test_set_1 = [10,20,23,29,3,32,33,35,37,41]

def TVSum_folder(video_reading_frequency=VIDEO_READING_FREQUENCY):
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device, train=TRAIN)

    # For generic video data folders
    # criterion = torch.nn.CrossEntropyLoss()  # Loss function (for classification)
    optimizer = optim.Adam(vqa.learn_user_preferences.parameters(), lr=0.001)  
    losses = []
    user_annotations = pickle.load(open('/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/user_annotations.pkl',"rb"))
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        if video_file != "sTEELN-vY30.mp4": #"sTEELN-vY30.mp4":
            continue
        optimizer.zero_grad()
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        video = VideoFileClip(video_path)
        total_frames = int(video.duration)*video_reading_frequency  # Number of seconds (since 1 fps)
        
        fps = video.fps  # Frames per second
        video_file_name = video_file.split('.')[0]
        timestamps = []
        
        for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=video_reading_frequency, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
            vqa.process_new_frame(frame)
            timestamps.append(i)

            # if i % 9 == 0 and i>0:
            #     vqa.answer_question("What is happening in this video?")
        
        # getting the user value for the relevant frames and segments
        frame_indices = [round(ts * fps) for ts in timestamps]  
        annotations_at_timestamps = user_annotations[video_file_name].iloc[frame_indices]
        frame_mean_values = annotations_at_timestamps['frame_mean']  
        mask_frames = torch.tensor(frame_mean_values>2.5, dtype=torch.bool)  

        # num_segments = (mask_frames.shape[0] - OVERLAP) // (SEGMENT_LENGTH - OVERLAP)  # Calculate the number of full segments
        # # Handle the case where the last segment is not full:
        # if (mask_frames.shape[0] - OVERLAP) % (SEGMENT_LENGTH - OVERLAP) != 0:
        #         num_segments += 1
        num_segments = len(vqa.segment_embeddings)
        mask_segments = torch.zeros(num_segments, dtype=torch.bool)
        start = 0
        for i in range(num_segments):
            end = min(start + SEGMENT_LENGTH, len(vqa.frame_embeddings))  # Handle edge cases where the last segment is smaller
            segment = mask_frames[start:end]
            num_true = segment.sum()
            num_false = segment.shape[0] - num_true
            mask_segments[i] = num_true > num_false
            start += (SEGMENT_LENGTH - OVERLAP)
        

        # path_frame_embeddings = f'/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/embeddings/languagebind_embs/frames/{video_file_name}/{VIDEO_READING_FREQUENCY}_fps'
        # Path(path_frame_embeddings).mkdir(parents=True, exist_ok=True)
        # vqa.save_emebeddings(vqa.frame_embeddings, path_frame_embeddings)
        # path_segment_embeddings = f'/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/embeddings/languagebind_embs/segments/{video_file_name}/{VIDEO_READING_FREQUENCY}_fps/{SEGMENT_LENGTH}_segment_{OVERLAP}_overlap'
        
        dpp_samples, L = vqa.answer_question("Person", video, video_reading_frequency)
        # plot_selections(video, video_reading_frequency, samples=dpp_samples, path='text.png')
        prob = dpp_probability(L, dpp_samples)
        reward = get_user_feedback_annotated(frame_mean_values, dpp_samples)
        loss = -reward * torch.log(prob)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        video.close()

    print("Done")

import re

def numerical_sort_key(filename):
    """Extracts the numerical part of the filename for sorting."""
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return filename  

def Avocado_folder():
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device, train=TRAIN)
    video_path = "/scratch3/kat049/user_studies/vids/isairas_2_cut/fps_2/frames"
    frame_names = [
    p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = sorted(frame_names, key=numerical_sort_key)
    
    for frame_path in frame_names:
        image_path = os.path.join(video_path, frame_path)
        img = Image.open(image_path)
        frame = np.array(img)
        vqa.process_new_frame(frame)

    cps_May_demo("avocados", video_path, frame_names, vqa)


def DARPA_folder(video_reading_frequency=VIDEO_READING_FREQUENCY):
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device, train=TRAIN)
    
    video_path =  '/scratch3/kat049/user_studies/vids/p17_fr.mp4'
    video = VideoFileClip(video_path)
    total_frames = int(video.duration)*video_reading_frequency  # Number of seconds (since 1 fps)

    fps = video.fps  # Frames per second
    timestamps = []
    frames = []
    for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=video_reading_frequency, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
        vqa.process_new_frame(frame)
        frames.append(frame)
        timestamps.append(i)
        
    dpp_samples, B = vqa.answer_question("Describe the video", video, video_reading_frequency)
    video.close()


def QVHighlights_folder():
    """Assumption: relevant windows = 1 (only one segment is enough). Relevant windows > 1 (need one segment from each window)"""
    from data_gen import read_json_file
    SEGMENT_LENGTH = 8
    VIDEO_READING_FREQUENCY = int(8/2) # a segment would be 2 seconds
    OVERLAP = 0
    VIDEO_FOLDER = '/scratch3/kat049/datasets/QVHighlights/videos'
    DEVICE = "cuda:2"
    TRAIN = False

    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=DEVICE, train=TRAIN)


    json_path = f'/scratch3/kat049/moment_detr/data/highlight_val_release.jsonl'
    json_data = read_json_file(json_path)
    json_df = pd.DataFrame(json_data)
    
    for _, row in tqdm.tqdm(json_df.iterrows(), total=len(json_df), desc="Processing Videos"):
        video_file_id = '_'.join(row.vid.split("_")[:-2])
        # if 'W9px1LFMICg' not in video_file_id:
        #     "frame 204600 - 600, segment:25575 - 75"
        #     continue
        video_file = video_file_id + ".mp4"
        
        if video_file not in video_files:
            continue
        
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        # sub_json_df = json_df[json_df.vid.str.startswith(video_file_id)]
        start_time =  row.vid.split("_")[-2]
        end_time = row.vid.split("_")[-1]
        video = VideoFileClip(video_path).subclipped(float(start_time), float(end_time))

        total_frames = int(video.duration)*VIDEO_READING_FREQUENCY

        try:
            assert row.duration == video.duration
        except AssertionError:
            print(f"Duration mismatch: {row.duration} != {video.duration} in {row.qid=}")

        for _, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=VIDEO_READING_FREQUENCY, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
            vqa.process_new_frame(frame)
        
        for v in ['frame_embeddings', 'segment_embeddings']:
            path_vision_embeddings = f'/scratch3/kat049/datasets/QVHighlights/val/freq{VIDEO_READING_FREQUENCY}_seg{SEGMENT_LENGTH}_overlap{OVERLAP}/qid{row.qid}_{v}.pt'
            Path(path_vision_embeddings).parent.mkdir(parents=True, exist_ok=True)
            embeddings = torch.stack(getattr(vqa, v)).detach()
            torch.save(embeddings, path_vision_embeddings)
        vqa.reset()
        # path_query_embeddings = f'/scratch3/kat049/datasets/QVHighlights/val/freq{VIDEO_READING_FREQUENCY}_seg{SEGMENT_LENGTH}_overlap{OVERLAP}/qid{row.qid}_query_embedding.pt'
        # embeddings = vqa.get_question_embedding(row.query)
        # torch.save(embeddings, path_query_embeddings)
        video.close()
        # print("Done")

def WildScenes_folder():
    VIDEO_NO = 'v-02'
    SEGMENT_LENGTH = 8
    OVERLAP = SEGMENT_LENGTH - 1
    IMAGE_FOLDER = f'/scratch3/kat049/datasets/WildScenes/WildScenes2D/{VIDEO_NO}/data/image'
    DEVICE = "cuda:2"
    TRAIN = False

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sorted_image_files = sorted(image_files, key=lambda x: int(re.match(r'(\d+)-', x).group(1)) if re.match(r'(\d+)-', x) else 0)
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=DEVICE, train=TRAIN)

    for _, file_name in tqdm.tqdm(enumerate(sorted_image_files), total=len(sorted_image_files), desc="Frames"):
        frame = np.array(Image.open(os.path.join(IMAGE_FOLDER, file_name)))
        vqa.process_new_frame(frame)
    
    for v in ['frame_embeddings', 'segment_embeddings']:
        path_vision_embeddings = f'/scratch3/kat049/datasets/WildScenes/WildScenes2D/{VIDEO_NO}/seg{SEGMENT_LENGTH}_overlap{OVERLAP}/{v}.pt'
        Path(path_vision_embeddings).parent.mkdir(parents=True, exist_ok=True)
        embeddings = torch.stack(getattr(vqa, v)).detach()
        torch.save(embeddings, path_vision_embeddings)
    vqa.reset()
    # path_query_embeddings = f'/scratch3/kat049/datasets/QVHighlights/val/freq{VIDEO_READING_FREQUENCY}_seg{SEGMENT_LENGTH}_overlap{OVERLAP}/qid{row.qid}_query_embedding.pt'
    # embeddings = vqa.get_question_embedding(row.query)
    # torch.save(embeddings, path_query_embeddings)

def cps_May_demo(query, video_path, frame_names, vqa):
    from tune_embeddings import prev_greedy, qvhighlights_topk_samples

    if query is None:
        query = 'avocados'
    frame_embeddings = torch.stack(getattr(vqa, 'frame_embeddings')).detach()
    query_embedding = vqa.get_question_embedding(query)

    num_items = 5
    do_L_norm = False
    
    indices_greedy, indices_simple = prev_greedy(frame_embeddings, query_embedding, num_items=num_items)

    # With DPP without training
    L_original = kernel_simple_batched(frame_embeddings[None, ...], query_embedding[None, ...])
    best_samples_original = qvhighlights_topk_samples(L_original, topk=10, do_L_norm=do_L_norm)

    # plot at end
    to_plot = (indices_greedy, best_samples_original) # best_samples_trained
    titles = ['Greedy (previous work)', 'Expected']
    max_columns = max(len(i) for i in to_plot)
    fig, axs = plt.subplots(len(to_plot), max_columns)
    for row, samples in enumerate(to_plot): 
        for col, sample in enumerate(samples):
            img = Image.open(os.path.join(video_path, frame_names[sample]))
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
        for col in range(col+1, max_columns):
            axs[row, col].axis('off')
        axs[row, 0].set_title(titles[row])
    fig.tight_layout()
    plt.suptitle(query)
    plt.savefig('proved.png')
    plt.close()

if __name__ == "__main__":
    DARPA_folder()
    # WildScenes_folder()


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
from dppy.finite_dpps import FiniteDPP


import os
import pickle


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

# standard
def get_user_feedback(samples, mask):
    # mask = [True,True,True,True, False, True, False, True,True]
    not_mask = [not elem for elem in mask]
    D_selected= torch.tensor(samples)[mask]
    D_ignored = torch.tensor(samples)[not_mask]
    # L[D_selected,:][:,D_selected]
    return D_selected, D_ignored

def get_user_feedback_annotated(frame_mean_values, samples):
    u = frame_mean_values.take(samples)
    median = u.median()
    if median > 3:
        return 1.0 # thumps up
    else:
        return -1.0 # thumps down

def kernel_diff(frame_embeddings, question_embedding):
    diff = frame_embeddings - question_embedding
    number_of_frames = diff.shape[0]
    L = torch.zeros((number_of_frames, number_of_frames))
    for i in range(number_of_frames):
        for j in range(number_of_frames):
            L[i,j] = (diff[i] @ diff[j].T) / ((diff[i]**2).sum() * (diff[j]**2).sum())
    diff2 = torch.sum(diff**2, dim=-1)
    L_prev = (diff @ diff.T) / (diff2 * diff2[:,None])
    L_new = (diff @ diff.T) / (torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True)) @ torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True)).T)

def kernel_simple(frame_embeddings, question_embedding):
    """Args:
        image_embeddings: Tensor of shape [n_images, 768]
        query_embedding: Tensor of shape [768]"""
    
    # Normalize embeddings
    image_embeddings = torch.nn.functional.normalize(frame_embeddings, dim=1)
    query_embedding = torch.nn.functional.normalize(question_embedding, dim=0)

    # Calculate quality scores (relevance to query)
    quality_scores = image_embeddings @ query_embedding
    
    # Compute similarity matrix
    similarity = image_embeddings @ image_embeddings.T

    # Create the L-ensemble kernel: L(i,j) = q_i * q_j * S(i,j)
    L_kernel = quality_scores[:, None] * quality_scores[None, :] * similarity

    # n = len(image_embeddings)
    # L_kernel = torch.zeros((n, n))    
    # for i in range(n):
    #     for j in range(n):
    #         L_kernel[i, j] = quality_scores[i] * quality_scores[j] * similarity[i, j]
    return L_kernel

def expected_num_frames(*, K=None, L=None):
    if K is None: K = torch.linalg.solve(L + torch.eye(len(L), device=L.device), L)
    else: assert L is None
    return K.trace()

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
        # torch.nn.init.eye_(self.linear.weight) # Access the Linear layer's weight
        # torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # W_sym = (self.linear.weight + self.linear.weight.t()) / 2
        # y = torch.matmul(x, W_sym.t()) + self.linear.bias # making sure this is symmetric
        y = self.linear (x)
        return y
    
class Incremental_VQA_LanguageBind():
    def __init__(self, segment_size=8, overlap=1, device='cpu', train=True):
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

        self.frame_emebeddings = []
        self.segment_embeddings = []

        # self.cluster_manager_frames = ClusterManager()
        # self.cluster_manager_segments = ClusterManager()

        self.dummy_text = [""]
        self.lambda_param = 0.01

        self.learn_user_preferences = UserAdaptation(input_size=768, device=self.device)

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

        self.frame_emebeddings.append(embeddings)
        self.frames.append(frame)


        if len(self.frames) == self.segment_size:
            self.process_segment()
            self.frames = self.frames[-self.overlap:]

    def process_segment(self):
        frames_tensor = torch.tensor(self.frames).to(self.device)  # Shape: (seq_len, height, width, 3)
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

        if len(self.frames) == self.segment_size:
            self.frames = self.frames[-self.overlap:]

    # def loss(self, user_mask, mask):
    #     FP = ((user_mask & ~mask).float()).mean()
    #     TP = ((user_mask & mask).float()).mean()
    #     FN = ((~user_mask & mask).float()).mean()
    #     TN = ((~user_mask & ~mask).float()).mean()
    
    def DPP_loss(K, sampled_set, mask):
        D_selected, D_ignored = get_user_feedback(samples=sampled_set, mask=mask)
        p_selected_in_sampled = K[D_selected,:][:,D_selected]

    def save_emebeddings(self, embeddings):
        embeddings = torch.stack(embeddings).detach()
        torch.save(embeddings, f'{self.embed_saving_path}/embeddings.pt')

    def answer_question(self, question, video = None, video_reading_frequency = 1):
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
        frame_embeddings = torch.stack(self.frame_emebeddings)
        
        # pdf, best_x = compare_embeds(question_embedding.detach(), frame_embeddings.detach())
        # pdf.argmax().item()
        # mask = torch.tensor(best_x, dtype=torch.bool)  

        
        # B = frame_embeddings*question_embedding
        # B = frame_embeddings * (0.5+ 0.5*(frame_embeddings @ question_embedding) / (frame_embeddings.norm(dim=-1) * question_embedding.norm(dim=-1)) )[:,None]
        # L = B @ B.T
        L = kernel_simple(frame_embeddings, question_embedding)
        L = L.detach().cpu()

        self.lambda_param = 1 / L.diag().max()
        
        dpp = FiniteDPP("likelihood", L = self.lambda_param * L)

        print('expected_num_frames =', expected_num_frames(L=L*self.lambda_param))
        samples = dpp.sample_exact()
        print('actual   num frames =', len(samples))
        samples.sort()
        plot_selections(video, video_reading_frequency, samples=samples, path='text.png')
        return
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

    def dpp_probability(self, B, samples):
        L = B @ B.T
        L = self.lambda_param * L
        L_S = L[samples, :][:, samples] # Extract submatrix
        try:
            prob = torch.det(L_S)
        except Exception as e:
            print(f"Error calculating determinant: {e}")
            prob = torch.tensor(1e-6) # To prevent log(0)
        return prob

def define_video_file_numbers():
    order = ['AwmHb44_ouw', '98MoyGZKHXc', 'J0nA4VgnoCo', 'gzDbaEs1Rlg', 'XzYM3PfTM4w', 'HT5vyqe0Xaw', 'sTEELN-vY30', 'vdmoEJ5YbrQ', 'xwqBXPGE9pQ', 'akI8YFjEmUw', 'i3wAGJaaktw', 'Bhxk-O1Y7Ho', '0tmA_C6XwfM', '3eYKfiOEJNs', 'xxdtq8mxegs', 'WG0MBPpPC6I', 'Hl-__g2gn_A', 'Yi4Ij2NM7U4', '37rzWOQsNIw', 'LRw_obCPUt0', 'cjibtmSLxQ4', 'b626MiF1ew4', 'XkqCExn6_Us', 'GsAD1KT1xo8', 'PJrm840pAUI', '91IHQYk1IQM', 'RBCABdttQmI', 'z_6gVvQb2d0', 'fWutDQy1nnY', '4wU_LUjG5Ic', 'VuWGsYPqAX8', 'JKpqYvAdIsw', 'xmEERLqJ2kU', 'byxOvuiIJV0', '_xMr-HKMfVA', 'WxtbjNsCQ8A', 'uGu_10sucQo', 'EE-bNr36nyA', 'Se3oxnaPsz0', 'oDXZc0tZe04', 'qqR6AEXwxoQ', 'EYqVtI9YWJA', 'eQu1rNs0an0', 'JgHubY5Vw3Y', 'iVt07TCkFM0', 'E11zDS9XGzg', 'NyBmCxDoHJU', 'kLxoNp-UchI', 'jcoYJXDG9sw', '-esJrBWj2d8', ]
    test_set_1 = [10,20,23,29,3,32,33,35,37,41]

def TVSum_folder():
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
        total_frames = int(video.duration)  # Number of seconds (since 1 fps)
        
        fps = video.fps  # Frames per second
        video_file_name = video_file.split('.')[0]
        timestamps = []
        
        for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=VIDEO_READING_FREQUENCY, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
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
            end = min(start + SEGMENT_LENGTH, len(vqa.frame_emebeddings))  # Handle edge cases where the last segment is smaller
            segment = mask_frames[start:end]
            num_true = segment.sum()
            num_false = segment.shape[0] - num_true
            mask_segments[i] = num_true > num_false
            start += (SEGMENT_LENGTH - OVERLAP)
        

        # path_frame_embeddings = f'/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/embeddings/languagebind_embs/frames/{video_file_name}/{VIDEO_READING_FREQUENCY}_fps'
        # Path(path_frame_embeddings).mkdir(parents=True, exist_ok=True)
        # vqa.save_emebeddings(vqa.frame_emebeddings, path_frame_embeddings)
        # path_segment_embeddings = f'/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/embeddings/languagebind_embs/segments/{video_file_name}/{VIDEO_READING_FREQUENCY}_fps/{SEGMENT_LENGTH}_segment_{OVERLAP}_overlap'
        
        dpp_samples, B = vqa.answer_question("Person", video, VIDEO_READING_FREQUENCY)
        prob = vqa.dpp_probability(B, dpp_samples)
        reward = get_user_feedback_annotated(frame_mean_values, dpp_samples)
        loss = -reward * torch.log(prob)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        video.close()

    print("Done")

def main():
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device, train=TRAIN)

    video_path =  '/scratch3/kat049/user_studies/vids/p17_fr.mp4'
    video = VideoFileClip(video_path)
    total_frames = int(video.duration)  # Number of seconds (since 1 fps)

    fps = video.fps  # Frames per second
    timestamps = []
    for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=VIDEO_READING_FREQUENCY, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
        vqa.process_new_frame(frame)
        timestamps.append(i)

    dpp_samples, B = vqa.answer_question("Describe the video", video, VIDEO_READING_FREQUENCY)

    video.close()

    print("Done")

if __name__ == "__main__":
    TVSum_folder()


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

import os
import pickle


SEGMENT_LENGTH = 8  # Segement length in frames
OVERLAP = 1  # Overlap between segments in frames
# --- Hyperparameters (Adjust these) ---
VIDEO_FOLDER = "/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/video" # Path to the folder containing videos

device = "cuda:3"


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

        self.learn_user_preferences = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
        ).to(self.device)
        torch.nn.init.eye_(self.learn_user_preferences[0].weight) # Access the Linear layer's weight
        torch.nn.init.zeros_(self.learn_user_preferences[0].bias)

    def process_new_frame(self, frame):
        """frame: np.ndarray"""
        inputs = {}
        inputs = {
            'image': to_device(self.modality_transform['image'](frame), self.device),
        }        

        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)
            
            # clustering 
            # full_embeddings = self.LanguageBindModel.modality_encoder['image'](inputs['image']['pixel_values'])[0]
            # self.cluster_manager_frames.add_tensor(full_embeddings)

        if self.train:
            embeddings = self.learn_user_preferences(embeddings['image'][0])
        else:
            with torch.no_grad():
                embeddings = self.learn_user_preferences(embeddings['image'][0])

        self.frame_emebeddings.append(embeddings)
        self.frames.append(frame)


        if len(self.frames) == self.segment_size:
            self.process_segment()
            self.frames = self.frames[-self.overlap:]

    def process_segment(self, vector_creation="cls_mean"):
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
            embeddings = self.LanguageBindModel(inputs)

            # forget what is not needed
            # full_embeddings = self.LanguageBindModel.modality_encoder['video'](inputs['video']['pixel_values'])[0]
            # self.cluster_manager_segments.add_tensor(full_embeddings)
        if self.train:
            embeddings = self.learn_user_preferences(embeddings['video'][0])
        else:
            with torch.no_grad():
                embeddings = self.learn_user_preferences(embeddings['video'][0])
        
            
        self.segment_embeddings.append(embeddings)

        if len(self.frames) == self.segment_size:
            self.frames = self.frames[-self.overlap:]

    def loss(self, user_mask, mask):
        FP = ((user_mask & ~mask).float()).mean()
        TP = ((user_mask & mask).float()).mean()
        FN = ((~user_mask & mask).float()).mean()
        TN = ((~user_mask & ~mask).float()).mean()
    

    def save_emebeddings(self, embeddings, path):
        '/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/embeddings/languagebind_embs/{video_file_name}/frame_embs.pt'
        torch.save(embeddings, path)

    def answer_question(self, question, output_format='images', user_mask_frames=None, user_mask_segments=None):
        # self.process_segment()

        language = question
        inputs = {}     
        inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)
        with torch.no_grad():
            question_embedding = self.LanguageBindModel(inputs)
        
        question_embedding  = question_embedding['language'][0]

        frame_embeddings = torch.stack(self.frame_emebeddings).detach()
        segment_embeddings = torch.stack(self.segment_embeddings).detach()

        pdf, best_x = compare_embeds(question_embedding, segment_embeddings)

        B = frame_embeddings*question_embedding
        L = B@B.T
        alpha = 1.5 / 31
        dpp = FiniteDPP("likelihood", L=alpha*L.cpu())
        dpp = DPP(segment_embeddings, question_embedding)


        pdf.argmax().item()
        mask = torch.tensor(best_x, dtype=torch.bool)  
        if output_format == 'video':
            pass
            # TODO combine the part of the video
        elif output_format == 'images':
            pdf, best_x = compare_embeds(question_embedding, frame_embeddings) 



def main():
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device)

    # criterion = torch.nn.CrossEntropyLoss()  # Loss function (for classification)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)  

    user_annotations = pickle.load(open('/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/user_annotations.pkl',"rb"))
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:       
        # if video_file != "sTEELN-vY30.mp4":
        #     continue
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        video = VideoFileClip(video_path)
        total_frames = int(video.duration)  # Number of seconds (since 1 fps)
        
        fps = video.fps  # Frames per second
        video_file_name = video_file.split('.')[0]
        timestamps = []
        
        for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=1, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
            vqa.process_new_frame(frame)
            timestamps.append(i)          
            # if i % 9 == 0 and i>0:
            #     vqa.answer_question("What is happening in this video?")
                    
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
        
        
        vqa.answer_question("What is happening in this video?", 'images', mask_frames, mask_segments)
        video.close()


if __name__ == "__main__":
    main()


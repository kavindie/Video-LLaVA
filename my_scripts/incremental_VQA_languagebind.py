import textwrap
import torch
import os
from collections import Counter, defaultdict
import tqdm
from moviepy import *
from PIL import Image
from scipy.optimize import minimize  # For optimization
from objective_functions import compare_embeds, solve_with_lasso
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

import os


SEGMENT_LENGTH = 8  # Segement length in frames
# --- Hyperparameters (Adjust these) ---
VIDEO_FOLDER = "/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/video" # Path to the folder containing videos

device = "cuda:3"



class Incremental_VQA_LanguageBind():
    def __init__(self, segment_size=8, overlap=1, device='cpu'):
        self.device = device
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

        self.dummy_text = [""]

    def process_new_frame(self, frame):
        """frame: np.ndarray"""
        language = self.dummy_text
        inputs = {}
        inputs = {
            'image': to_device(self.modality_transform['image'](frame), self.device),
        }        
        inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)
        
        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)

        self.frame_emebeddings.append(embeddings[0])
        self.frames.append(frame)

        if len(self.frames) == self.segment_size:
            self.process_segment()
            self.frames = self.frames[-self.overlap:]

    def process_segment(self, vector_creation="cls_mean"):
        frames_tensor = torch.tensor(self.frames).to(self.device)  # Shape: (seq_len, height, width, 3)
        language = self.dummy_text
        inputs= {}
        inputs = {
            'video': to_device(self.modality_transform['video'](frames_tensor), self.device),
        }        
        inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)
        
        with torch.no_grad():
            embeddings = self.LanguageBindModel(inputs)
            
        self.segment_embeddings.append(embeddings[0])

        if len(self.frames) == self.segment_size:
            self.frames = self.frames[-self.overlap:]


    def save_emebeddings(self, embeddings, path):
        torch.save(embeddings, path)

    def answer_question(self, question):
        language = question
        inputs = {}     
        inputs['language'] = to_device(self.tokenizer(language, max_length=77, padding='max_length',
                                                    truncation=True, return_tensors='pt'), self.device)
        with torch.no_grad():
            question_embeddings = self.LanguageBindModel(inputs)
        
        frame_embeddings = torch.cat(self.frame_emebeddings)
        segment_embeddings = torch.cat(self.segment_embeddings)


        pdf, best_x = compare_embeds(question_embeddings, segment_embeddings)
        pdf, best_x = compare_embeds(question_embeddings, frame_embeddings)
        pdf.argmax().item()
        mask = torch.tensor(best_x, dtype=torch.bool)  
        

        self.process_segment()

        # TODO answer the question
        pass

def main():
    vqa = Incremental_VQA_LanguageBind(device=device)

    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        video = VideoFileClip(video_path)
        total_frames = int(video.duration)  # Number of seconds (since 1 fps)
        for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=1, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
            vqa.process_new_frame(frame)
            if i % 9 == 0 and i>0:
                vqa.answer_question("What is happening in this video?")

        video.close()


if __name__ == "__main__":
    main()


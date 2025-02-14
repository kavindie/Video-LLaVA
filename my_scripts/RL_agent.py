import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from collections import defaultdict
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image
import re
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# --- Hyperparameters (Adjust these) ---
FRAME_HISTORY_LENGTH = 5  # Number of previous frames to include in state
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor for RL
EPSILON = 0.1  # Exploration rate
NUM_EPISODES = 100
BATCH_SIZE = 16  # For experience replay (if used)
CONCEPT_EXTRACTION_MODEL_PATH = "path/to/your/concept_extraction_model.pth" # Path to your concept extraction model
VIDEO_FOLDER = "/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/video" # Path to the folder containing videos
METRIC_WEIGHTS = {"SC": 0.4, "SP": 0.3, "QC": 0.2, "D": 0.1} # Weights for the metrics
NUM_CONCEPTS = 1000 # Number of concepts (adjust to your model)

cuda = "cuda:3"

ConceptExtractor = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
ConceptExtractor.to(cuda)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16).to(cuda)

# # --- Concept Extraction (Replace with your actual model) ---
# class ConceptExtractor(nn.Module):
#     def __init__(self):
#         super(ConceptExtractor, self).__init__()
#         # Define your model architecture here (e.g., ResNet, etc.)
#         self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#         self.model.fc = nn.Linear(self.model.fc.in_features, 1000) # Example: 1000 concepts

#     def forward(self, frames):
#         # frames shape: (batch, seq_len, 3, 224, 224)
#         batch_size, seq_len, c, h, w = frames.shape
#         frames = frames.view(-1, c, h, w) # Reshape for the model
#         features = self.model(frames) # Shape: (batch*seq_len, num_concepts)
#         features = features.view(batch_size, seq_len, -1) # Reshape back
#         return features


# concept_extractor = ConceptExtractor().to(f"{cuda}" if torch.cuda.is_available() else "cpu")
# concept_extractor.load_state_dict(torch.load(CONCEPT_EXTRACTION_MODEL_PATH)) # Load your trained model
# concept_extractor.eval()



# --- State Representation ---
def create_state(frame_features, summary_features):

    if summary_features is None:
      summary_features = torch.zeros_like(frame_features)  # Initialize if no summary yet

    state = torch.cat([frame_features, summary_features], dim=-1) # Concatenate features
    return state

# --- Metrics ---

def calculate_sc(current_concepts, all_concepts):
    intersection = np.intersect1d(current_concepts, all_concepts)
    return len(intersection) / len(all_concepts) if len(all_concepts) > 0 else 0

def calculate_sp(current_prominences, original_prominences):
    total_original_prominence = sum(original_prominences.values())
    if total_original_prominence == 0:
        return 0
    
    total_min_prominence = 0
    for concept, prominence in current_prominences.items():
        total_min_prominence += min(prominence, original_prominences.get(concept, 0))
    return total_min_prominence / total_original_prominence

def calculate_qc(current_concepts):
    return len(current_concepts)

def calculate_distance(summary_features, original_features):
    # Use cosine similarity as distance
    summary_features = summary_features.mean(dim=0) # Average over frames
    original_features = original_features.mean(dim=0) # Average over frames

    if torch.all(summary_features == 0) or torch.all(original_features == 0):
        return 1.0 # Max distance if either vector is all zeros
    
    similarity = torch.cosine_similarity(summary_features, original_features, dim=0)
    distance = 1 - similarity
    return distance.item()


# --- RL Agent ---
class Agent(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(Agent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1) # Output probabilities for each action
        )

    def forward(self, state):
        return self.fc(state)


def get_frame_features(video_folder, video_file, concept_extractor):
    video_path = os.path.join(video_folder, video_file)
    prompt = [
        "USER: <image> Provide a list of concepts, separated by commas, present in the image. Avoid speculation or interpretation. The list can have one or many items. Pay attention to objects, colors, events etc. Describe as many concepts as you can. ASSISTANT:"
    ]
    prompt = [
        "USER: <image> Provide a factual description of this image. Avoid speculation or interpretation. ASSISTANT:"
    ]
    
    frames = []
    all_concepts = set()
    original_prominences = defaultdict(float)
    video = VideoFileClip(video_path)

    interval = 25  #??

    # Get timestamps at which to extract frames
    timestamps = list(range(0, int(video.duration), 25))

    # total_frames = int(video.duration)  # Number of seconds (since 1 fps)
    #for i, frame in tqdm(enumerate(video.iter_frames(fps=1, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
        
    for i, t in tqdm.tqdm(enumerate(timestamps), total=len(timestamps), desc="Extracting Frames"):
        if i > 15:  # Limit to 15 frames
            break
        frame = video.get_frame(t)
        frames.append(frame)
        pil_image = Image.fromarray(frame)
        inputs = processor(text=prompt, images=pil_image, padding=True, return_tensors="pt")
        inputs = {k: v.to(cuda) for k, v in inputs.items()}
        # Generate
        generate_ids = concept_extractor.generate(**inputs, max_new_tokens=1000)
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        assistant_responses = []
        matches = re.findall(r"ASSISTANT:\s*(.+?)(?=(?:USER:|ASSISTANT:|$))", response[0], re.DOTALL)  # Improved regex
        for match in matches:
            assistant_responses.append(match.strip())
        
        img_description = ", ".join(assistant_responses) # Extract concepts from the response

        prompt_llm = f"""Find the key concepts in the given paragraph. Avoid speculation or interpretation. Give the output as a comma separated list of key concepts. 
            Paragraph: {img_description}
            Output:         
        """
        inputs = tokenizer(prompt_llm, return_tensors="pt").to(cuda)
        generate_ids = llm_model.generate(
            inputs.input_ids,
        )
        generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        prompt_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        output_text = generated_text[prompt_length:]
        concept_list = [concept.strip() for concept in output_text.split(",")]
        for concept_id in concept_list:
            all_concepts.add(concept_id)
            original_prominences[concept_id] += 1 # Increment prominence


    return all_concepts, original_prominences, frames


# --- Training Loop ---
def train(agent, concept_extractor, video_folder, num_episodes):
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for episode in range(num_episodes):
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        for video_file in video_files:
            all_concepts, original_prominences, frames = get_frame_features(video_folder, video_file, concept_extractor)
            all_concepts = list(all_concepts)  # Convert to list for indexing
            num_frames = len(frames)
            frames_tensor = torch.tensor(frames).to(f"{cuda}" if torch.cuda.is_available() else "cpu")  # Shape: (1, seq_len, 3, 224, 224)
            with torch.no_grad():
                prompt = [
                    "USER: <video> Provide a list of concepts present in the video. ASSISTANT:"
                ]
                inputs = processor(text=prompt, videos=frames_tensor, padding=True, return_tensors="pt")
                inputs = {k: v.to(cuda) for k, v in inputs.items()}
                generate_ids = concept_extractor.generate(**inputs, max_new_tokens=500)
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # original_features = concept_extractor(frames_tensor)  # Shape: (1, seq_len, num_concepts)

            summary_features_list = []  # Store features of selected frames
            selected_frames = []
            total_reward = 0

            for t in range(num_frames):
                frame_tensor = frames_tensor[:, t, :, :, :].unsqueeze(1)
                with torch.no_grad():
                    frame_features = concept_extractor(frame_tensor)
                state = create_state(frame_features, torch.stack(summary_features_list) if summary_features_list else torch.zeros(1,0,NUM_CONCEPTS).to(f"{cuda}" if torch.cuda.is_available() else "cpu")) #Provide empty tensor if no summary yet

                action_probs = agent(state)
                action = torch.multinomial(action_probs, 1).item()  # Sample action

                current_concepts = set()
                current_prominences = defaultdict(float)

                if action == 1:  # If the frame is selected
                    selected_frames.append(t)
                    summary_features_list.append(frame_features)

                    with torch.no_grad():
                        selected_frame_features = concept_extractor(frame_tensor)
                        selected_concepts = selected_frame_features.argmax(dim=-1).tolist()[0]
                        for concept_id in selected_concepts:
                            current_concepts.add(concept_id)
                            current_prominences[concept_id] += 1
                    current_concepts = list(current_concepts)  # Convert to list


                # Calculate reward (NOW CORRECTED)
                sc = calculate_sc(current_concepts, all_concepts)
                sp = calculate_sp(current_prominences, original_prominences)
                qc = calculate_qc(current_concepts)
                distance = calculate_distance(torch.stack(summary_features_list) if summary_features_list else torch.zeros(1,0,NUM_CONCEPTS).to(f"{cuda}" if torch.cuda.is_available() else "cpu"), original_features) # Distance to the entire original video

                reward = (METRIC_WEIGHTS["SC"] * sc +
                          METRIC_WEIGHTS["SP"] * sp +
                          METRIC_WEIGHTS["QC"] * qc -
                          METRIC_WEIGHTS["D"] * distance)

                total_reward += reward

                # Backpropagation (Corrected)
                optimizer.zero_grad()
                loss = -torch.log(action_probs[0, action]) * reward # Negative because we want to maximize reward
                loss.backward()
                optimizer.step()

            print(f"Episode: {episode}, Video: {video_file}, Total Reward: {total_reward}")

# --- Initialize and Train ---
input_dim = NUM_CONCEPTS * 2 # Input dimension to the agent (frame features + summary features)
num_actions = 2  # Select or don't select
agent = Agent(input_dim, num_actions).to(f"{cuda}" if torch.cuda.is_available() else "cpu")
train(agent, ConceptExtractor, VIDEO_FOLDER, NUM_EPISODES)
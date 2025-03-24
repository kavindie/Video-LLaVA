from math import prod
import random
import itertools
from collections.abc import Iterable, Collection
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from incremental_VQA_languagebind import UserAdaptation
from torch.nn.utils.rnn import pad_sequence
from dpp_utils import *
import math
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

SEGMENT_LENGTH = 8
VIDEO_READING_FREQUENCY = int(8/2) # a segment would be 2 seconds
OVERLAP = 0
VIDEO_FOLDER = '/scratch3/kat049/datasets/QVHighlights/videos'
DEVICE = "cuda:2"
TRAIN = False

class VideoSaliencyDataset(Dataset):
    def __init__(self, json_file, embedding_dir):
        """
        Args:
            json_file (string): Path to the JSON file containing video data.
            embedding_dir (string): Directory containing frame embeddings (.pt files).
        """
        self.data = []
        with open(json_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item['qid']
        
        saliency_scores = torch.tensor(item['saliency_scores'])

        # Load embeddings for the relevant qid
        vision_embedding_path = os.path.join(self.embedding_dir, f"qid{qid}_segment_embeddings.pt")
        query_embedding_path = os.path.join(self.embedding_dir, f"qid{qid}_query_embedding.pt")
        try:
            clip_embeddings = torch.load(vision_embedding_path)
            query_embedding = torch.load(query_embedding_path)
        except FileNotFoundError:
            # print(f"Warning: Embedding file not found for qid: {qid} segment or query embeddings.pt")
            return None # return None and handle in the dataloader.

        return {
            'qid': qid,
            'duration': item['duration'],
            'query': item['query'],
            'query_embedding': query_embedding,
            'vid': item['vid'],
            'relevant_windows': torch.tensor(item['relevant_windows']),
            'relevant_clip_ids': torch.tensor(item['relevant_clip_ids']),
            'clip_embeddings': clip_embeddings,
            'saliency_scores': torch.tensor(item['saliency_scores'])
        }

def collate_fn(batch):
    """
    Custom collate function to handle None values in the batch.
    """
    batch = [item for item in batch if item is not None] #Remove None from the batch
    if not batch: # if the batch is empty, return None
        return None

    collat_dict = {}
    for k in ['qid', 'duration', 'query', 'vid']:
        collat_dict[k] = [item[k] for item in batch]
        # qids = [item['qid'] for item in batch]
    collat_dict['query_embedding'] = torch.stack([item['query_embedding'] for item in batch])
    for k in ['query_embedding', 'relevant_windows', 'relevant_clip_ids', 'clip_embeddings', 'saliency_scores']:
        collat_dict[k] = pad_sequence([item[k] for item in batch], batch_first=True, padding_value=-1)
        #torch.stack([item[k] for item in batch])
       
    return collat_dict

# Example Usage:
json_file = f'/scratch3/kat049/moment_detr/data/highlight_val_release.jsonl'
embedding_dir = f'/scratch3/kat049/datasets/QVHighlights/val/freq{VIDEO_READING_FREQUENCY}_seg{SEGMENT_LENGTH}_overlap{OVERLAP}'  
full_dataset = VideoSaliencyDataset(json_file, embedding_dir)
# sample = full_dataset[0]  # Get the first sample (index 0)
# Split dataset into training and testing (e.g., 80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)  # No shuffle for testing

input_size = 768  # Adjust to your embedding size
output_size = input_size
model = UserAdaptation(input_size, output_size, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000


def dpp_logprob_from_pos_neg(*, L=None, K=None, pos=None, neg=None, obs=None):
    """
    Compute the log probability of a DPP given the kernel L and a subset of selected items.
    Including pos, not including neg.
    Args:
        L (torch.Tensor): Likelihood matrix of shape (..., n, n).
        K (torch.Tensor): Marginal Kernel matrix of shape (..., n, n).
        Provide 2 of the 3:
            pos: Iterable of indices of selected items.
            neg: Iterable of indices of unselected items.
            obs: Iterable of indices of observed items.
    Returns:
        logprob (torch.Tensor): Log probability of the subset, shape (...).
    """
    if K is None:
        K = calc_K_from_L(L)
    
    if   obs is None: obs = set(pos) | set(neg)
    elif neg is None: neg = set(obs) - set(pos)
    # elif pos is None: pos = set(obs) - set(neg)  # Not needed
    obs = list(obs)
    neg = list(neg)
    
    K_obs = K[..., obs, :][..., :, obs].clone()
    K_obs[..., neg, :] *= -1
    K_obs[..., neg, :][..., :, neg] += torch.eye(len(neg), device=K_obs.device)
    logprob = torch.logdet(K_obs)
    return logprob

def dpp_logprob_from_exact(*, L, pos, logdet_normalizer=0):
    """
    Compute the log probability of a DPP given the kernel L and a subset of selected items.
    Including pos, not including neg.
    Args:
        L (torch.Tensor): Likelihood matrix of shape (..., n, n).
        Provide 2 of the 3:
            pos: Iterable of indices of selected items.
    Returns:
        logprob (torch.Tensor): Log probability of the subset, shape (...).
    """
    pos = list(pos)
    L_pos = L[..., pos, :][..., :, pos]
    logprob = torch.logdet(L_pos) -logdet_normalizer
    return logprob

def pos_from_groups(groups, max_n=10):
    """
    Generate positive and negative indices from groups.
    Args:
        groups: Iterable of Collection of indices. One element is selected from each group, the rest are unselected. 
    Returns:
        Iterable of:
            pos: Collection of positive indices.
            neg: Collection of negative indices.
    """
    n = prod(len(group) for group in groups)
    if n > max_n:
        for _ in range(max_n):
            yield {random.choice(group) for group in groups}
    else:
        yield from itertools.product(*groups)


def dpp_logprob_from_groups(L, batched_groups, max_n=10):
    dpp_logprobs = []
    for L_group, group in zip(L, batched_groups):
        dpp_logprobs.append(torch.stack([dpp_logprob_from_exact(L=L_group, pos=pos) for pos in pos_from_groups(group, max_n=max_n)]).mean())

    return torch.stack(dpp_logprobs)


def groups_from_relevant_windows(relevant_windows):
    SEGMENT_LENGTH_SECONDS = 2
    batched_groups = []
    for batch in relevant_windows:
        groups = []
        for window in batch:
            start_time, end_time = window
            if start_time < 0 or end_time < 0:
                continue
            group = range(int(start_time/SEGMENT_LENGTH_SECONDS), int(end_time/SEGMENT_LENGTH_SECONDS))
            groups.append(group)
        batched_groups.append(groups)
    return batched_groups


def qvhighlights_loss(L, batch):
    batched_groups = groups_from_relevant_windows(batch['relevant_windows'])
    logprob = dpp_logprob_from_groups(L, batched_groups, max_n = 20)
    return -logprob.mean()


# --- Training Loop ---
max_loss = math.inf
train_loss_values = []
test_loss_values = []
fig, axes = plt.subplots(1, 2)
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        if batch is None:
            continue
        clip_embeddings = batch['clip_embeddings'].to(DEVICE)
        query_embedding = batch['query_embedding'].to(DEVICE)

        optimizer.zero_grad()

        predicted_clip_embeddings = model(clip_embeddings)
        predicted_query_embedding = model(query_embedding)
        L = kernel_simple_batched(predicted_clip_embeddings, predicted_query_embedding) 
        loss = qvhighlights_loss(L, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    average_train_loss = train_loss / len(train_dataloader)
    train_loss_values.append(average_train_loss)
    axes[0].plot(train_loss_values, '.-r')
    plt.savefig('loss.png')
    
    # --- Test Loop ---
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in test_dataloader:
            if batch is None:
                continue
            clip_embeddings = batch['clip_embeddings'].to(DEVICE)
            query_embedding = batch['query_embedding'].to(DEVICE)

            predicted_clip_embeddings = model(clip_embeddings)
            predicted_query_embedding = model(query_embedding)
            L = kernel_simple_batched(predicted_clip_embeddings, predicted_query_embedding) 
            loss = qvhighlights_loss(L, batch)
            test_loss += loss.item()

        if test_loss < max_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            max_loss = test_loss

    average_test_loss = test_loss / len(test_dataloader)
    test_loss_values.append(average_test_loss)
    axes[1].plot(test_loss_values, '*-b')
    plt.savefig('loss.png')

    # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}")

print("Training finished!")
    
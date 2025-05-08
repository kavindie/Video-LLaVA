from math import prod
import random
import itertools
from collections.abc import Iterable, Collection
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
from torch.nn.utils.rnn import pad_sequence
import math
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from moviepy import *

from incremental_VQA_languagebind import UserAdaptation, Incremental_VQA_LanguageBind
from data_gen import get_video_frames
from dpp_utils import *
from plot_utils import plot_accuracy_loss



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

def qvhighlights_loss_2(L, batch, topk=1, do_L_norm=False):
    """L = [..., n, n]"""
    batch_size = batch['relevant_clip_ids'].shape[0]
    batch_loss = []
    # K = calc_K_from_L(L)
    for b in range(batch_size):
        best_samples = qvhighlights_topk_samples(L[b][None, ...], topk=topk, do_L_norm=do_L_norm)
        relevant_clip_ids = batch['relevant_clip_ids'][b]
        saliency_score = batch['saliency_scores'][b]
        # mask = [s in best_samples for s in relevant_clip_ids]
        # saliency_score[mask]

        D_ignored = []
        potential_samples = []
        potential_scores = []

        if best_samples is None:
            D_selected = []
            D_ignored = []
            loss = -dpp_logprob_from_exact(L=L[b], pos=D_selected) + dpp_logprob_from_exact(L=L[b], pos=D_ignored)
            batch_loss.append(loss)
        
        for s in best_samples:
            if s not in relevant_clip_ids:
                D_ignored.append(s)
            else:
                index_s = list(relevant_clip_ids).index(s)
                score_s = saliency_score[index_s].to(torch.float).mean()
                potential_samples.append(s)
                potential_scores.append(score_s)
            
        if len(D_ignored) == len(best_samples):
            # None of the selected samples in the relevant_clip_ids
            # get the loss for D_selected = [] and D_ignored = best_samples
            break

        correct_sample_index = torch.stack(potential_scores).topk(k=1).indices
        selected_sample = potential_samples[correct_sample_index]
        
        D_selected = [selected_sample]
        for s in potential_samples:
            if s != selected_sample:
                D_ignored.append(s)
        D_ignored.sort()
        # get the loss for D_selected = [] and D_ignored = best_samples
        
        # D_si = D_selected + D_ignored

        # K_DS = K[b][:, D_selected][D_selected, :]
        # K_DI = K[b][:, D_ignored][D_ignored, :]
        # K_DIS = K[b][:, D_si][D_si, :]

        # For now
        loss = -dpp_logprob_from_exact(L=L[b], pos=D_selected) + dpp_logprob_from_exact(L=L[b], pos=D_ignored)
        batch_loss.append(loss)
    
    return torch.stack(batch_loss).mean()
        
       
        
def qvhighlights_loss(L, batch):
    batched_groups = groups_from_relevant_windows(batch['relevant_windows'])
    logprob = dpp_logprob_from_groups(L, batched_groups, max_n=20)
    return -logprob.mean()

def qvhighlights_topk_samples(L, topk=20, do_L_norm=False):
    # Currently only handles one batch
    # L = [1, n, n ]
    if len(L.shape) == 3:
        L = L[0]
    
    max_prob = 0
    best_samples = None
    for k in range(topk):
        _, _, samples, prob_samples = process_single_dpp(L, do_L_norm=do_L_norm)
        if prob_samples > max_prob:
            max_prob = prob_samples
            best_samples = samples
    return best_samples

def multi_group_accuracy(selected_items, groups):
    """
    Calculates accuracy for multiple groups, penalizing multiple selections and incorrect items.

    Args:
        selected_items: A list of items selected by the user.
        groups: A list of lists, where each inner list represents a group.

    Returns:
        A score between 0.0 and 1.0, reflecting the accuracy.
    """

    num_groups = len(groups)
    if num_groups == 0:
        return 1.0 if not selected_items else 0.0 # if no groups, empty selection is perfect

    group_representation = [False] * num_groups  # Track if each group is represented
    # correct_selections = 0
    incorrect_selections = 0
    selection_count_per_group = [0] * num_groups # track selection count per group.
    length_per_group = [len(group) for group in groups]

    for item in selected_items:
        found_in_group = False
        for group_index, group in enumerate(groups):
            if item in group:
                found_in_group = True
                if selection_count_per_group[group_index] == 0:
                    group_representation[group_index] = True
                    # correct_selections += 1
                selection_count_per_group[group_index] +=1
                break
        if not found_in_group:
            incorrect_selections += 1

    group_representation_score_w_multi_selection = sum([a/b if b > 0 else 0 for a, b in zip(length_per_group, selection_count_per_group)])/num_groups
    selection_accuracy = 1 - incorrect_selections/len(selected_items) if len(selected_items) > 0 else 0
    final_score = selection_accuracy*group_representation_score_w_multi_selection
    return final_score

    group_representation_score = sum(group_representation) / num_groups if num_groups > 0 else 1.0 # 1 if all groups represented
    multi_selection_penalty = 0

    for c in selection_count_per_group:
      if c > 1:
        multi_selection_penalty += c -1 # penalty for each extra selection in a group.

    incorrect_penalty = incorrect_selections / len(selected_items) if len(selected_items) > 0 else 0;
    multi_selection_penalty = multi_selection_penalty / len(selected_items) if len(selected_items) > 0 else 0;

    final_score = group_representation_score - multi_selection_penalty - incorrect_penalty

    return max(0.0, min(1.0, final_score)) # ensure the score is in the 0-1 range

def qvhighlights_accuracy(L, batch):
    "Not handling empty sets"
    batched_groups = groups_from_relevant_windows(batch['relevant_windows'])
    batched_groups = batched_groups[0] # because in the test there is only one group
    best_samples = qvhighlights_topk_samples(L)

    total_accuracy = multi_group_accuracy(best_samples, batched_groups)
    return total_accuracy

def init(json_file, embedding_dir, train_batch_size=4, test_batch_size=4):
    # Example Usage:
    full_dataset = VideoSaliencyDataset(json_file, embedding_dir)
    # sample = full_dataset[0]  # Get the first sample (index 0)
    # Split dataset into training and testing (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)  # No shuffle for testing

    return train_dataloader, test_dataloader

def train_loop(num_epochs, model, optimizer, train_dataloader, test_dataloader, device, load_checkpoint=None):
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
            clip_embeddings = batch['clip_embeddings'].to(device)
            query_embedding = batch['query_embedding'].to(device)

            optimizer.zero_grad()

            predicted_clip_embeddings = model(clip_embeddings)
            predicted_query_embedding = model(query_embedding)
            L = kernel_simple_batched(predicted_clip_embeddings, predicted_query_embedding)
            loss = qvhighlights_loss_2(L, batch, topk=20, do_L_norm=False)
            # loss = qvhighlights_loss(L, batch)
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
                clip_embeddings = batch['clip_embeddings'].to(device)
                query_embedding = batch['query_embedding'].to(device)

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
    return train_loss_values, test_loss_values

def eval_loop(model, test_dataloader, device):
    model.eval()  # Set model to evaluation mode
    baseline_acc, baseline_loss = [], []
    trained_acc, trained_loss = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            if batch is None:
                continue
            clip_embeddings = batch['clip_embeddings'].to(device)
            query_embedding = batch['query_embedding'].to(device)

            L = kernel_simple_batched(clip_embeddings, query_embedding) 
            accuracy = qvhighlights_accuracy(L, batch)
            baseline_acc.append(accuracy)
            baseline_loss.append(qvhighlights_loss(L, batch))

            predicted_clip_embeddings = model(clip_embeddings)
            predicted_query_embedding = model(query_embedding)
            
            L = kernel_simple_batched(predicted_clip_embeddings, predicted_query_embedding) 
            accuracy = qvhighlights_accuracy(L, batch)
            trained_acc.append(accuracy)
            trained_loss.append(qvhighlights_loss(L, batch))
    plot_accuracy_loss(baseline_acc, trained_acc, torch.stack(baseline_loss).detach().cpu(), torch.stack(trained_loss).detach().cpu())

def prev_greedy(frame_embeddings, query_embedding, num_items=10):
    import sys
    sys.path.append('/scratch3/kat049/user_studies')
    from graph_algo import select_items

    v = frame_embeddings @ query_embedding.T
    s = torch.softmax(v, dim=0)
    s_flattened = s.view(-1)

    # greedy approach
    xf = v
    probabilities_xf = s_flattened
    ff = frame_embeddings @ frame_embeddings.T
    normalized_ff = (ff - ff.min()) / (ff.max() - ff.min())
    normalized_ff.fill_diagonal_(float('-inf'))
    probabilities_ff = torch.nn.functional.softmax(normalized_ff, dim=1)

    indices_s = torch.tensor(select_items(probabilities_xf, probabilities_ff, num_items=num_items))
    indices_s = indices_s.sort().values.tolist()
    return indices_s, probabilities_xf.topk(num_items).indices.sort().values.tolist()

def prove_my_point(VIDEO_FOLDER, VIDEO_READING_FREQUENCY, SEGMENT_LENGTH, OVERLAP, DEVICE, model, test_dataloader, embedding_dir):
    num_items = 5
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=DEVICE, train=False)
    
    with torch.no_grad():
        for batch in test_dataloader:
            if batch is None:
                continue

            _, frames = get_video_frames(batch['vid'][0], VIDEO_FOLDER, VIDEO_READING_FREQUENCY)
            do_L_norm = False

            query = batch['query'][0] #'woman'

            # Original way
            frame_embeddings = torch.load(os.path.join(embedding_dir, f"qid{batch['qid'][0]}_frame_embeddings.pt")).to(DEVICE)
            query_embedding = vqa.get_question_embedding(query).to(DEVICE)
            indices_greedy, indices_simple = prev_greedy(frame_embeddings, query_embedding, num_items=num_items)

            # With DPP without training
            L_original = kernel_simple_batched(frame_embeddings[None, ...], query_embedding[None, ...])
            best_samples_original = qvhighlights_topk_samples(L_original, do_L_norm=do_L_norm)
            
            # best_samples_original = greedy_map_dpp_fast_torch(L_original[0])
            # if best_samples_original is None:
            #     input("No samples selected by DPP")

            # WIth DPP with training - not great
            predicted_frame_embeddings = model(frame_embeddings[None, ...])
            predicted_query_embedding = model(query_embedding[None, ...])
            L_trained = kernel_simple_batched(predicted_frame_embeddings, predicted_query_embedding)
            best_samples_trained = qvhighlights_topk_samples(L_trained, do_L_norm=do_L_norm)
  

            # plot at end
            to_plot = (indices_simple, indices_greedy, best_samples_original) # best_samples_trained
            titles = ['Simple sim based', 'Initial Greedy (previous work)', 'DPP - original']
            max_columns = max(len(i) for i in to_plot)
            fig, axs = plt.subplots(len(to_plot), max_columns)
            for row, samples in enumerate(to_plot): 
                for col, sample in enumerate(samples):
                    axs[row, col].imshow(frames[sample])
                    axs[row, col].axis('off')
                for col in range(col+1, max_columns):
                    axs[row, col].axis('off')
                axs[row, 0].set_title(titles[row])
            fig.tight_layout()
            plt.suptitle(query)
            plt.savefig('proved.png')
            
def check_dataset():
    SEGMENT_LENGTH = 8
    VIDEO_READING_FREQUENCY = int(8/2) # a segment would be 2 seconds
    OVERLAP = 0
    VIDEO_FOLDER = '/scratch3/kat049/datasets/QVHighlights/videos'
    DEVICE = "cuda:2"
    json_file = f'/scratch3/kat049/moment_detr/data/highlight_val_release.jsonl'
    embedding_dir = f'/scratch3/kat049/datasets/QVHighlights/val/freq{VIDEO_READING_FREQUENCY}_seg{SEGMENT_LENGTH}_overlap{OVERLAP}'
    input_size = 768  # Adjust to your embedding size
    output_size = input_size
    model = UserAdaptation(input_size, output_size, DEVICE).to(DEVICE)
    train_dataloader, test_dataloader = init(json_file, embedding_dir, train_batch_size=1, test_batch_size=1)
    for batch in tqdm(train_dataloader):
        if batch is None:
            continue
        
        full_vid_file_name = batch['vid'][0]
        video_file_id = '_'.join(full_vid_file_name.split("_")[:-2])
        video_file = video_file_id + ".mp4"    
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        start_time =  full_vid_file_name.split("_")[-2]
        end_time = full_vid_file_name.split("_")[-1]
        video = VideoFileClip(video_path).subclipped(float(start_time), float(end_time))

        relevant_windows = batch['relevant_windows']
        subclips = []
        if relevant_windows.shape[1] > 1:
            for window in relevant_windows[0]:
                start_time, end_time = window
                subclip = video.subclipped(float(start_time), float(end_time))
                subclips.append(subclip)
                black_frame = ColorClip(size=video.size, color=(0, 0, 0), duration=0.5)
                subclips.append(black_frame)
            
            if subclips:
                final_clip = concatenate_videoclips(subclips)
                output_path = os.path.join('/scratch3/kat049/Video-LLaVA/my_scripts/QVHighlights_testing', f"{video_file_id}_{relevant_windows.shape[1]}_relwindows_{batch['query'][0]}.mp4")
                final_clip.write_videofile(output_path)
                final_clip.close()

def main():
    SEGMENT_LENGTH = 8
    VIDEO_READING_FREQUENCY = int(8/2) # a segment would be 2 seconds
    OVERLAP = 0
    # VIDEO_FOLDER = '/scratch3/kat049/datasets/QVHighlights/videos'
    DEVICE = "cuda:2"
    TRAIN = False
    json_file = f'/scratch3/kat049/moment_detr/data/highlight_val_release.jsonl'
    embedding_dir = f'/scratch3/kat049/datasets/QVHighlights/val/freq{VIDEO_READING_FREQUENCY}_seg{SEGMENT_LENGTH}_overlap{OVERLAP}'
    num_epochs = 1000

    input_size = 768  # Adjust to your embedding size
    output_size = input_size
    model = UserAdaptation(input_size, output_size, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if TRAIN:
        train_dataloader, test_dataloader = init(json_file, embedding_dir, train_batch_size=4, test_batch_size=4)
        train_loss_values, test_loss_values = train_loop(num_epochs, model, optimizer, train_dataloader, test_dataloader, DEVICE)
    else:
        _, test_dataloader = init(json_file, embedding_dir, train_batch_size=4, test_batch_size=1)
        import sys
        sys.path.insert(0,'/scratch3/kat049/moment_detr')
        from standalone_eval.eval import eval_moment_retrieval


        CHCEKPOINT_PATH = '/scratch3/kat049/Video-LLaVA/my_scripts/best_model_1.pth'
        VIDEO_FOLDER = '/scratch3/kat049/datasets/QVHighlights/videos'
        model.load_state_dict(torch.load(CHCEKPOINT_PATH, weights_only=True))
        prove_my_point(VIDEO_FOLDER, VIDEO_READING_FREQUENCY, SEGMENT_LENGTH, OVERLAP, DEVICE, model, test_dataloader, embedding_dir)
        eval_loop(model, test_dataloader, DEVICE)
         
if __name__ == '__main__':
    main()
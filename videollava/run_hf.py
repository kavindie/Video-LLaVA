from PIL import Image
import requests
import numpy as np
import av
import glob
import os
import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips

from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
model.to('cuda:3')
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# prompt = "USER: <video>Why is this video funny? ASSISTANT:"
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
# container = av.open(video_path)

# # sample uniformly 8 frames from the video
# total_frames = container.streams.video[0].frames
# indices = np.arange(0, total_frames, total_frames / 8).astype(int)
# clip = read_video_pyav(container, indices)

# inputs = processor(text=prompt, videos=clip, return_tensors="pt")
# inputs = {k: v.to('cuda:3') for k, v in inputs.items()}

# # Generate
# generate_ids = model.generate(**inputs)
# print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# Generate from images and videos mix
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# url = "https://cdn.shopify.com/s/files/1/0353/7521/8732/files/INIKA-environ-pollution-blog.jpg?v=1596995181"
# image = Image.open(requests.get(url, stream=True).raw)


# path_to_folder = '/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/images/_xMr-HKMfVA'
# image_paths = sorted(glob.glob(os.path.join(path_to_folder, "frame_*.jpg")))


# for image_path in tqdm.tqdm(image_paths[(1417+1421+1419):], desc="Processing Images", unit="image"):  # Add description and unit
#     image = Image.open(image_path)
#     # inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")
#     inputs = processor(text=prompt, images=image, padding=True, return_tensors="pt")
#     inputs = {k: v.to('cuda:2') for k, v in inputs.items()}

#     # Generate
#     generate_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.2, top_p=0.9)
#     # print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

prompt = [
    # "USER: <image> Provide a factual description of this image. Avoid speculation or interpretation.  ASSISTANT:",
    "<image>"
]

clip = VideoFileClip('/scratch3/kat049/datasets/Epic-Kitchens/test/P01/P01_11.MP4')
print(f'Duration: {clip.duration/60} minutes with a frame rate of {clip.fps}')
clip = clip.subclip(0, 60) # Consider only the subset of the video

frames = []
frame_count = 0
for frame in clip.iter_frames():
    if frame_count % (60*4) != 0:
        frame_count += 1
        continue
    frames.append(frame)
    pil_image = Image.fromarray(frame)
    # pil_image.save(f'/scratch3/kat049/datasets/Epic-Kitchens/test/P01/Images/frame_{frame_count}.jpg')
    inputs = processor(text=prompt, images=pil_image, padding=True, return_tensors="pt")
    inputs = {k: v.to('cuda:3') for k, v in inputs.items()}
    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=200)
    # print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    frame_count += 1
frames = np.array(frames)
frames_pruned = frames[::60*4]

prompt = [
    "<video> Give me a description of this video",
]
inputs = processor(text=prompt, videos=frames, padding=True, return_tensors="pt")
inputs = {k: v.to('cuda:3') for k, v in inputs.items()}
generate_ids = model.generate(**inputs, max_new_tokens=500)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))


prompt = [
    "USER: <image> Provide a factual description of this image. Avoid speculation or interpretation.  ASSISTANT:",
]
inputs = processor(text=prompt, images=frame, padding=True, return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}
generate_ids = model.generate(**inputs, max_new_tokens=500)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
clip.close()


# History of terminal
# E = model.language_model.model.embed_tokens.weight.T
# H = model.language_model.lm_head.weight
# P = torch.linalg.lstsq(E.T, H)
# P.solution
# P.residuals  # ?

# patches = pixel_values_images.reshape(1,3,16,14,16,14).permute([0,2,4,1,3,5])  #.reshape(-1,3,14,14)
# image_features  # exists
# h = hidden_states[:, -1:, :, None]
# e = P.solution @ h
# im_logit = image_features @ e

# pos = im_logit.squeeze().topk(10)
# neg = im_logit.squeeze().neg().topk(10)
# for i in range(10):
#     j = pos.indices.squeeze()[i]
#     torchvision.utils.save_image(patches.reshape(-1,3,14,14)[j], f'stefan/out/+{i} ({j}).jpg')
# for i in range(10):
#     j = neg.indices.squeeze()[i]
#     torchvision.utils.save_image(patches.reshape(-1,3,14,14)[j], f'stefan/out/-{i} ({j}).jpg')





# # # TODO remove after
# path_to_save = '/scratch3/kat049/datasets/Epic-Kitchens/test/P01/frame_embeddings.pt'
# image_embeddings.append(hidden_states)
# with open(path_to_save, 'wb') as f:
#     torch.save(torch.stack(image_embeddings), f)
# # #
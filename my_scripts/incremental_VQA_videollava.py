import textwrap
import torch
import os
from collections import Counter, defaultdict
import tqdm
from moviepy import *
from PIL import Image
from scipy.optimize import minimize  # For optimization
from objective_functions import compare_embeds, solve_with_lasso


from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollava.eval.video.run_inference_video_qa import get_model_output

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEGMENT_LENGTH = 8  # Segement length in frames
# --- Hyperparameters (Adjust these) ---
VIDEO_FOLDER = "/scratch3/kat049/STVT/STVT/STVT/datasets/datasets/datasets/ydata-tvsum50-v1_1/video" # Path to the folder containing videos

device = "cuda:3"

# use .venv
# from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
# self.embedding_model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
# self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")


# To use the video-llava. There is an error about the device
# def get_model_output_updated(model, video_processor, tokenizer, video, qs, device):
#     if model.config.mm_use_im_start_end:
#         qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
#     else:
#         qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

#     conv_mode = "llava_v1"
    

#     conv = conv_templates[conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()


#     video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(device)
#     # print(video_tensor.shape)
#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=[video_tensor],
#             do_sample=False,
#             temperature=0.0,
#             max_new_tokens=1024,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria])

#     input_token_len = input_ids.shape[1]
#     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
#     if n_diff_input_output > 0:
#         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
#     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
#     outputs = outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs = outputs[:-len(stop_str)]
#     outputs = outputs.strip()
#     print(outputs)
#     return outputs


class Incremental_VQA_VideoLLaVA():
    def __init__(self, segment_size=8, overlap=1, device='cpu'):
        model_path = 'LanguageBind/Video-LLaVA-7B'
        model_base = None
        model_name = get_model_name_from_path(model_path)
        
        self.tokenizer, self.embedding_model, self.processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        self.embedding_model = self.embedding_model.to(device)
        self.conv_mode = "llava_v1"
        
        # output = get_model_output_updated(model, processor['video'], tokenizer, video_path, question, device)

        # with .venv
        # self.embedding_model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
        # self.embedding_model.to(self.device)
        # self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

        if segment_size > 8:
            raise ValueError("Segment size should be less than or equal to 8")
        self.segment_size = segment_size

        if overlap > segment_size or overlap < 0:
            raise ValueError("Overlap should be between 0 and segment_size")
        self.overlap = overlap    

        self.frames = []

        self.frame_emebeddings = []
        self.segment_embeddings = []

        self.frame_full_emebeddings = []
        self.segment_full_embeddings = []

        self.dummy_text = [""]

    def process_new_frame(self, frame, vector_creation="CLS"):
        """frame: np.ndarray"""


        with torch.no_grad():
            image = Image.fromarray(frame)
            inputs = self.processor(images=image, text=self.dummy_text, return_tensors="pt")
            
            image_output = self.embedding_model.get_image_features(inputs['pixel_values_images'].to(self.device), -2, 'full')  # after the multimodal projection

            self.frame_full_emebeddings.append(image_output)

            if vector_creation == 'CLS':
                img_vector = image_output[:, 0, :]  # Shape: [1, 4096]
            elif vector_creation == 'mean':
                img_vector = image_output[:, 1:, :].mean(dim=1)
            elif vector_creation == 'max':  
                img_vector = image_output[:, 1:, :].max(dim=1)[0]
            else:
                raise ValueError("vector_creation should be one of 'CLS', 'mean', 'max'")
            
            self.frame_emebeddings.append(img_vector)

        self.frames.append(frame)

        if len(self.frames) == self.segment_size:
            self.process_segment()
            self.frames = self.frames[-self.overlap:]

    def process_segment(self, vector_creation="cls_mean"):
        with torch.no_grad():
            frames_tensor = torch.tensor(self.frames).to(self.device)  # Shape: (seq_len, height, width, 3)
            inputs = self.processor(text=self.dummy_text, videos=frames_tensor, padding=True, return_tensors="pt")
            video_output = self.embedding_model.get_video_features(inputs['pixel_values_videos'].to(self.device), -2)[0] #mutlimodal projection
            
            self.segment_full_embeddings.append(video_output)

            if vector_creation == 'cls_mean':
                cls_tokens = video_output[:, 0, :]  # Shape: [15, 4096] - get CLS token from each frame
                video_vector = cls_tokens.mean(dim=0)  # Shape: [4096] - mean across frames
            elif vector_creation == 'mean':
                video_vector = video_output.mean(dim=[0, 1]) 
            else:
                raise ValueError("vector_creation should be one of 'cls_mean', 'mean'")

            self.segment_embeddings.append(video_vector)

        if len(self.frames) == self.segment_size:
            self.frames = self.frames[-self.overlap:]


    def save_emebeddings(self, embeddings, path):
        torch.save(embeddings, path)

    def answer_question(self, question):
        frame_embeddings = torch.cat(self.frame_emebeddings)
        segment_embeddings = torch.cat(self.segment_embeddings)
        question_embedding = self.processor(text=question, padding=True, return_tensors="pt")

        self.embedding_model.language_model.get_input_embeddings()

        inputs = {k: v.to(device) for k, v in question_embedding.items()}
        generate_ids = self.embedding_model.generate(**inputs, max_new_tokens=200)

        # num_image_tokens = (height // self.patch_size) * (
        #         width // self.patch_size
        # ) + self.num_additional_image_tokens
        # num_video_tokens = num_image_tokens * num_frames
        # prompt_strings = []
        # for sample in text:
        #     sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
        #     sample = sample.replace(self.video_token, self.video_token * num_video_tokens)
        #     prompt_strings.append(sample)
        # prompt_strings = [question]
        # text_inputs = self.processor.tokenizer(
        #     prompt_strings,
        #     return_tensors=return_tensors,
        #     padding=padding,
        #     truncation=truncation,
        #     max_length=max_length,
        # )
        # data.update(text_inputs)
        # generate
        # output_attentions = False
        # output_hidden_states = False
        # return_dict = True
        # vision_feature_layer = -2
        # vision_feature_select_strategy = 'full'

        # for images
        # video_features = ?
        # n_video_features = video_features.shape[0] * video_features.shape[1]
        # n_video_tokens = n_video_features
        # special_image_mask = (input_ids == self.config.video_token_index).unsqueeze(-1)
        # special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        # video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
        # inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)

        # attention_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:3')
        # position_ids =tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        # past_key_values = DynamiCache()
        # cache_position = tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:3')

        # outputs = self.embedding_model.language_model(
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=True,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     return_dict=True,
        #     cache_position=cache_position,
        #     num_logits_to_keep=1,
        # )

        # logits = outputs[0]

        # loss = None

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # VideoLlavaCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     image_hidden_states=image_features if pixel_values_images is not None else None,
        #     video_hidden_states=video_features if pixel_values_videos is not None else None,
        # )



        pdf, best_x = compare_embeds(self.segment_embeddings[0], frame_embeddings)
        pdf.argmax().item()
        mask = torch.tensor(best_x, dtype=torch.bool)  
        

        self.process_segment()

        # TODO answer the question
        pass

def main():
    vqa = Incremental_VQA_VideoLLaVA(device=device)

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


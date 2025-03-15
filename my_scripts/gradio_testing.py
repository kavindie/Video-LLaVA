import time
import gradio as gr
import os
from incremental_VQA_languagebind import Incremental_VQA_LanguageBind, VIDEO_FOLDER, SEGMENT_LENGTH, OVERLAP, device, TRAIN, VIDEO_READING_FREQUENCY
from moviepy import *
import tqdm


MODEL_INFER = True

vid_path2show = os.path.join(VIDEO_FOLDER, 'sTEELN-vY30.mp4')
video_file = "sTEELN-vY30.mp4"
video_path = os.path.join(VIDEO_FOLDER, video_file)
video = VideoFileClip(video_path)
total_frames = int(video.duration)  # Number of seconds (since 1 fps)

fps = video.fps  # Frames per second
video_file_name = video_file.split('.')[0]

vqa = None
if  MODEL_INFER:
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device, train=TRAIN)
    for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=VIDEO_READING_FREQUENCY, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
        vqa.process_new_frame(frame)
    

def user_feedback_func(text, page_title):
    global vqa, video
    print(text, page_title)
    if MODEL_INFER:
        dpp_samples, _ = vqa.answer_question(text)
        frames = [video.get_frame(s*VIDEO_READING_FREQUENCY) for s in dpp_samples] #TODO need to check if this gets the correct image index
    
    else:
        frames = ['/scratch3/kat049/Video-LLaVA/000000039769.jpg']*16
    # top_images, times , top_video_path, chatbot = models.queryLanguageBindImageVideoText(text, chatbot, top_k_video=top_k_video, k_images = k_images,vid_num=0)
    return frames

def user_feedback_page(title="Help me get better"):
  with gr.Blocks(title=title) as page:
    # with gr.Column():
        gr.Markdown(f"# {title}")
        gr.Markdown("""
            The system will try to pick the best images that matches your query. Some of the images might not be relevant. Some might be repetitive. 
            Please select the **minimum** number of images that helped you answer your question. 
            If none of the images answered your query, please select the button, `None'.  
        """)
        with gr.Column(variant="panel"):
            text_input = gr.Text(label='User Query')
            #video = gr.Video(label="Video Output", render=False, show_download_button=False)
            gallery = gr.Gallery(label="Image Output", render=False, columns=[5], rows=[4], object_fit="contain", height="auto", preview=True, show_download_button=False, show_share_button=False)
            #chatbot = gr.Chatbot(label='Chat Output', render=False)
            with gr.Row():
                #clear_button = gr.ClearButton([text_input, chatbot])
                submit_btn = gr.Button("Submit", variant="primary")
                submit_btn.click(user_feedback_func, [text_input], [gallery])
                text_input.submit(user_feedback_func, [text_input], [gallery])


        with gr.Row(variant="panel"):
            gallery.render()

        gr.Markdown("""
            Below you can find the original video of what happened. 
            It will take a couple of seconds to load.
            """)
        with gr.Row():
            with gr.Column(scale=1, visible=True):
                gr.Markdown("Adding space", visible=False)
            with gr.Column(scale=1):
                gr.Video(value=f"{vid_path2show}", show_label=False, show_download_button=False)
            with gr.Column(scale=1, visible=True):
                gr.Markdown("Adding space", visible=False)


  return page

if __name__ == "__main__":
    urls = {}
    pages = [user_feedback_page]
    pages = [page() for page in pages]  # make 'em!
    for demo in pages:
        demo.queue()
        demo.launch(prevent_thread_lock=True)
        urls[demo.title] = demo.local_url

    while all(demo.is_running for demo in pages):
        time.sleep(1)

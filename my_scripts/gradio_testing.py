import time
import gradio as gr
import os
from moviepy import *
import tqdm
from incremental_VQA_languagebind import VIDEO_FOLDER


MODEL_INFER = False

vid_path2show = os.path.join(VIDEO_FOLDER, 'sTEELN-vY30.mp4')
video_file = "sTEELN-vY30.mp4"
video_path = os.path.join(VIDEO_FOLDER, video_file)
video = VideoFileClip(video_path)
total_frames = int(video.duration)  # Number of seconds (since 1 fps)

fps = video.fps  # Frames per second
video_file_name = video_file.split('.')[0]
vqa = None
if  MODEL_INFER:
    from incremental_VQA_languagebind import Incremental_VQA_LanguageBind, SEGMENT_LENGTH, OVERLAP, device, TRAIN, VIDEO_READING_FREQUENCY
    vqa = Incremental_VQA_LanguageBind(segment_size=SEGMENT_LENGTH, overlap=OVERLAP, device=device, train=TRAIN, trained_output_size=None)
    for i, frame in tqdm.tqdm(enumerate(video.iter_frames(fps=VIDEO_READING_FREQUENCY, dtype="uint8")), total=total_frames, desc="Extracting Frames"):
        vqa.process_new_frame(frame)
    

def user_feedback_func(text, page_title):
    global vqa, video, frames
    print(text, page_title)
    if MODEL_INFER:
        dpp_samples, _ = vqa.answer_question(text)
        frames = [video.get_frame(s*VIDEO_READING_FREQUENCY) for s in dpp_samples] #TODO need to check if this gets the correct image index
    
    else:
        # import torchvision
        # img_path = '/scratch3/kat049/Video-LLaVA/my_scripts/000000039769.jpg' #TODO need to fix this new path issue
        # frames = []
        # for i in range(16):
        #     frames.append(
        #         (torchvision.transforms.functional.adjust_contrast(torchvision.io.read_image(img_path), i/10)).numpy()
        #     ) 
        frames = ['/scratch3/kat049/Video-LLaVA/my_scripts/000000039769.jpg' ]*4 + ['/scratch3/kat049/Video-LLaVA/my_scripts/text_kavi.png']*4 + ['/scratch3/kat049/Video-LLaVA/my_scripts/text.png']*4 + ['/scratch3/kat049/Video-LLaVA/my_scripts/text_sim.png'] * 8
        # TODO need to fix this new path issue
    # top_images, times , top_video_path, chatbot = models.queryLanguageBindImageVideoText(text, chatbot, top_k_video=top_k_video, k_images = k_images,vid_num=0)
    return frames, frames

def add_selected_images(imgs, selected_imgs, selected, selected_indices):
    selected = int(selected)

    if selected_imgs is None:
        selected_imgs = [imgs[selected]]
        selected_indices = [selected]
    else:
        if selected in selected_indices:
            return selected_imgs, selected_imgs, selected_indices
        selected_imgs.append(imgs[selected])
        selected_indices.append(selected)

    return selected_imgs, selected_imgs, selected_indices

def get_select_index(evt: gr.SelectData):
    return evt.index

def deselect_images():
    return gr.Gallery(selected_index=None)

def done_selecting_func(selected_indices, imgs, selected_imgs, ignored_imgs):
    global vqa #TODO calculate loss
    if selected_indices is None:
        ignored_imgs = imgs
    else:
        ignored_imgs = [x for i, x in enumerate(imgs) if i not in selected_indices]
    return [], selected_imgs, ignored_imgs

def clear_all():
    return None, None, None, None, None, None, None, None, None, None

def clear_with_submit():
    return None, None, None, None, None, None

def user_feedback_page(title="Help me get better"):
  with gr.Blocks(title=title) as page:
        imgs = gr.State()
        selected_imgs = gr.State()
        ignored_imgs = gr.State()
        selected_indices = gr.State()
        selected = gr.State()
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
                clear_button = gr.ClearButton([text_input])
                submit_btn = gr.Button("Submit", variant="primary")
                
                text_input.submit(user_feedback_func, [text_input], [gallery, imgs])
                
        with gr.Row(variant="panel"):
            gallery.render()

        with gr.Column(variant="panel"):
            with gr.Row():
                # selected = gr.Number(show_label=False, interactive=False)
                approve_btn = gr.Button("Add Selection")
            # deselect_button = gr.Button("Deselect")
            done_selecting_button = gr.Button("Done Selecting")

        gallery.select(get_select_index, None, selected)

        with gr.Column(variant="panel"):
            with gr.Row():
                selected_gallery = gr.Gallery(label="Selected Images", render=False, columns=[5], rows=[4], object_fit="contain", height="auto", preview=False, show_download_button=False, show_share_button=False)
            with gr.Row():
                ignored_gallery = gr.Gallery(label="Ignored Images", render=False, columns=[5], rows=[4], object_fit="contain", height="auto", preview=False, show_download_button=False, show_share_button=False)
        
        with gr.Row(variant="panel"):
            selected_gallery.render()
            ignored_gallery.render()
        
        approve_btn.click(add_selected_images, [imgs, selected_imgs, selected, selected_indices],  [selected_imgs, selected_gallery, selected_indices])
        # deselect_button.click(deselect_images, None, gallery)
        done_selecting_button.click(done_selecting_func, [selected_indices, imgs, selected_imgs, ignored_imgs],[gallery, selected_gallery, ignored_gallery])
        clear_button.click(clear_all, None, [text_input, gallery, selected_gallery, ignored_gallery, imgs, selected_imgs, ignored_imgs, selected_indices, selected])
        submit_btn.click(user_feedback_func, [text_input], [gallery, imgs]).then(clear_with_submit, None, [selected_gallery, ignored_gallery, selected_imgs, ignored_imgs, selected_indices, selected])

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

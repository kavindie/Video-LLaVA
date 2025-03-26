import json
import yt_dlp
from moviepy import *
import os 


def read_json_file(path):
    """
    Read a file and return a list of JSON objects
    Args:
        path (str): The path to the file
    Returns:
        list: A list of JSON objects
    """
    data = []
    with open(path, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                # Handle the error or skip the line
                continue
    return data

def vid_to_link(vid):
    *youtube_id, start_time, end_time = vid.split("_")
    youtube_id = '_'.join(youtube_id)
    return youtube_id, f"https://www.youtube.com/watch?v={youtube_id}"


def download_video(url, output_folder="/scratch3/kat049/datasets/QVHighlights/val", youtube_id=None):
    output_filename = f"{output_folder}/{youtube_id}.mp4"

    ydl_opts = {
        "format": "best",
        "outtmpl": output_filename,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Downloaded: {output_filename}")
    except Exception as e:
        print(f"Error downloading video: {e}")
        return

# def trim_video(output_filename, start_time=None, end_time=None):
    # # Trim the video if start_time and end_time are provided
    # if start_time is not None and end_time is not None:
    #     trimmed_filename = f"trimmed_{output_filename}"
    #     clip = VideoFileClip(output_filename).subclip(float(start_time), float(end_time))
    #     clip.write_videofile(trimmed_filename, codec="libx264")

    #     print(f"Trimmed video saved as: {trimmed_filename}")

def get_video_frames(full_vid_file_name, video_folder, video_reading_frequency):
    # Get video for plotting
    video_file_id = '_'.join(full_vid_file_name.split("_")[:-2])
    video_file = video_file_id + ".mp4"    
    video_path = os.path.join(video_folder, video_file)
    start_time =  full_vid_file_name.split("_")[-2]
    end_time = full_vid_file_name.split("_")[-1]
    video = VideoFileClip(video_path).subclipped(float(start_time), float(end_time))

    frames = list(video.iter_frames(fps=video_reading_frequency, dtype="uint8"))
    return video, frames
    
if __name__ == "__main__":
    for i in ['val', 'test']:
        json_path = f'/scratch3/kat049/moment_detr/data/highlight_{i}_release.jsonl'
        saving_folder = f'/scratch3/kat049/datasets/QVHighlights/videos'
        processed_videos = [f for f in os.listdir(saving_folder) if f.endswith(('.mp4'))]
        # processed_videos = os.listdir(saving_folder)

        data = read_json_file(json_path)
        processed_data = []
        
        for i, d in enumerate(data):
            vid = d['vid']
            youtube_id, url = vid_to_link(vid)
            if youtube_id in processed_data or f'{youtube_id}.mp4' in processed_videos:
                continue
            print (f"Processing video {youtube_id}")
            processed_data.append(youtube_id)
            download_video(url, youtube_id=youtube_id, output_folder=saving_folder)
            

"""['qid', 'query', 'duration', 'vid', 'relevant_clip_ids', 'saliency_scores', 'relevant_windows']"""
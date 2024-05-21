import openai
from youtube_transcript_api import YouTubeTranscriptApi
import os

def get_youtube_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript_list

def convert_transcript_to_text(transcript_list):

    text = ""

    for titles in transcript_list:
        text = text + f"[{titles['start']}] " + f"{titles['text']} "

    return text

def get_video_id_from_url(video_url):
    if "?v=" in video_url:

        start_pos = video_url.find("?v=") + 3
        video_id = video_url[start_pos:]
    else:
        raise ValueError("Invalid Video URL")

    return video_id


if __name__ == '__main__':

    test_url = "https://www.youtube.com/watch?v=v0YCyZ8l-gQ"

    transcript = get_youtube_transcript(get_video_id_from_url(test_url))

    text = convert_transcript_to_text(transcript)

    print(text)

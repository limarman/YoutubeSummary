import dataclasses
import json
from datetime import datetime

import mistune
import notionfier
from notionfier import NotionPageBuilder, NotionfierRender
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import os
from notion_client import Client
from markdown_it import MarkdownIt
from markdown_it.token import Token

from pytube import YouTube

import argparse


NOTION_AUTH_FILE = "notion.txt"
OPENAI_KEY_FILE = "openai.txt"

def connect_notion_client():

    if not os.path.exists(NOTION_AUTH_FILE):
        raise ValueError("The notion authentication file could not be found!")

    with open(NOTION_AUTH_FILE, 'r') as file:
        auth = file.read()

    notion_client = Client(auth=auth)

    return notion_client

def connect_openai_client():

    if not os.path.exists(OPENAI_KEY_FILE):
        raise ValueError("The openai key file could not be found!")

    with open(OPENAI_KEY_FILE, 'r') as file:
        api_key = file.read()

    openai_client = openai.Client(api_key=api_key)

    return openai_client


def get_youtube_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript_list

def convert_transcript_to_text(transcript_list, timestamp_freq_secs=10):

    text = ""

    time_unit_counter = 0

    for titles in transcript_list:
        if float(titles['start']) > time_unit_counter * timestamp_freq_secs:
            time_unit_counter += 1
            timestamp = f" [{float(titles['start']):.0f}]"
        else:
            timestamp = ""

        text = text + timestamp + f" {titles['text']}"

    return text

def get_video_id_from_url(video_url):
    if "?v=" in video_url:

        start_pos = video_url.find("?v=") + 3
        video_id = video_url[start_pos:]
    else:
        raise ValueError("Invalid Video URL")

    return video_id

"""
def markdown_to_notion_blocks(markdown_text: str):
    md = MarkdownIt()
    tokens = md.parse(markdown_text)

    # parse tokens
    def parse_tokens(tokens):
        blocks = []

        for token in tokens:
            if token.type == 'heading_open':
                level = int(token.tag[1])
                content_token = next(tokens)
                blocks.append({
                    "object": "block",
                    "type": f"heading_{level}",
                    f"heading_{level}": {
                        "rich_text": [{
                            "type": "text",
                            "text": {
                                "content": content_token.content
                            }
                        }]
                    }
                })
            elif token.type == 'paragraph_open':
                content_token = next(tokens)
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {
                                "content": content_token.content
                            }
                        }]
                    }
                })
            elif token.type == 'blockquote_open':
                content_token = next(tokens)
                blocks.append({
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [{
                            "type": "text",
                            "text": {
                                "content": content_token.content
                            }
                        }]
                    }
                })
            #elif token.type == 'fence':
            #    blocks.append({
            #        "object": "block",
            #        "type": "code",
            #        "code": {
            #            "text": [{
            #                "type": "text",
            #                "text": {
            #                    "content": token.content
            #                }
            #            }]
            #        }
            #    })
            elif token.type == 'list_item_open':
                list_type = 'bulleted_list_item' if token.tag == 'li' else 'numbered_list_item'
                content_token = next(tokens)
                blocks.append({
                    "object": "block",
                    "type": list_type,
                    list_type: {
                        "rich_text": [{
                            "type": "text",
                            "text": {
                                "content": content_token.content
                            }
                        }]
                    }
                })
            elif token.type == 'hr':
                blocks.append({
                    "object": "block",
                    "type": "divider",
                })
            # Add more cases as needed for lists, images, etc.
        return blocks

    blocks = parse_tokens(iter(tokens))

    return blocks
"""

def markdown_to_notion_blocks(markdown_text: str):

    builder = NotionPageBuilder(token="EMPTY")

    notion_objects = builder.parse_content(markdown_text)

    blocks = [
            dataclasses.asdict(x, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
            for x in notion_objects
            ]

    return blocks

def create_notion_page(markdown_text, notion_client, page_properties, header_image_url=None):

    blocks = markdown_to_notion_blocks(markdown_text)

    if header_image_url:

        # Create the image block
        image_block = {
            "object": "block",
            "type": "image",
            "image": {
                "type": "external",
                "external": {
                    "url": header_image_url
                }
            }
        }

        # Insert the image block at the beginning of the blocks list
        blocks.insert(0, image_block)

    new_page = {
        "parent": {"type": "database_id", "database_id": DATABASE_ID},
        "properties": page_properties,
        'children': blocks
    }

    notion_client.pages.create(**new_page)


def extract_openai_response(completion):

    return completion.choices[0].message.content


def query_openai_model(user_content, system_message, openai_client: openai.Client, temperature=0.2, model='gpt-4o'):

    if isinstance(user_content, str):
        user_content = [user_content]

    messages = [{'role': 'user', 'content': content} for content in user_content]
    messages.insert(0, {'role': 'system', 'content': system_message})

    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return completion

def get_user_and_system_message(file):

    if not os.path.exists(file):
        raise ValueError("The file does not exist!")

    with open(file, 'r') as prompt_file:
        messages = prompt_file.read()

    if "SYSTEM:" not in messages or "PROMPT:" not in messages:
        raise ValueError("Invalid prompt file!")

    system_start = messages.find('SYSTEM:') + 7
    prompt_start = messages.find('PROMPT:') + 7

    system_message = messages[system_start:prompt_start-7].strip()
    user_message = messages[prompt_start:].strip()

    return system_message, user_message


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-url", "--video-url", required=True, help="Url to the Youtube video you want to summarize.")
    parser.add_argument("-dbid", "--database-id", default="abe67fb8a1294451a174c8fd61d0ea52", help="The id of the database you want to add the summary to.")
    parser.add_argument("-ui", "--user-info", default="", help="Additional information that is going to be written in the header of the page.")
    parser.add_argument("-reup", "--reupload", action='store_true', help="Uploads the summary output from latest_completion.json to notion instead of regenerating.")

    args = parser.parse_args()

    DATABASE_ID = args.database_id
    URL = args.video_url

    if not args.reupload:

        #url = "https://www.youtube.com/watch?v=v0YCyZ8l-gQ"

        # -------------------------------------------------------------------
        # Getting the transcript
        # -------------------------------------------------------------------

        print("PROGRESS: Retrieving the youtube video transcript...")

        transcript = get_youtube_transcript(get_video_id_from_url(URL))

        text = convert_transcript_to_text(transcript)

        # -------------------------------------------------------------------
        # Query open AI for summary
        # -------------------------------------------------------------------

        print("PROGRESS: Querying OpenAI for summary...")

        system_message, user_message = get_user_and_system_message("prompt.txt")

        openai_client = connect_openai_client()

        completion = query_openai_model([text, user_message], system_message, openai_client)

        with open('latest_completion.json', 'w') as response:
            json.dump(completion.to_dict(), response, indent=4)

        output = extract_openai_response(completion)

    else:
        with open("latest_completion.json", 'r') as latest:
            data = json.load(latest)

        output = data['choices'][0]['message']['content']

    # ---------------------------------------------------------------------------------------
    # Upload the data to notion
    # --------------------------------------------------------------------------------------

    print("PROGRESS: Uploading data to Notion...")

    # 1. Parse the title and the markdown
    if not "--summary-start--" in output:
        raise ValueError("Model output does not conform with specification!")

    summary_start = output.find('--summary-start--')

    title = output[:summary_start]
    summary = output[summary_start+17:]

    yt = YouTube(URL)

    # 2. Upload to notion
    notion_client = connect_notion_client()

    page_properties = {
        "Name": {"type": "title", "title": [{ "type": "text", "text": { "content": title } }]},
        "Video URL":{"type": "url", "url": URL},
        "Info": {"type": "rich_text", "rich_text": [{ "type": "text", "text": { "content": args.user_info } }]},
        "Creation Date": {"type": "date", "date": {"start": datetime.now().date().isoformat()}}
        }

    create_notion_page(summary, notion_client=notion_client, page_properties=page_properties,
                       header_image_url=yt.thumbnail_url)


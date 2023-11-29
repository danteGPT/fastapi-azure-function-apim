from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from pydantic import BaseModel
from celery import Celery
import re
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import uvicorn
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from pytrends.request import TrendReq
import requests
import json
from datetime import timedelta
import subprocess
import sys
import shutil
import openai
import logging
import tempfile

CELERY_BROKER_URL = "redis://localhost:6379/0"  # Redis URL for the broker
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"  # Redis URL for the result backend

celery = Celery("worker", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
# Initialize Celery
celery_app = Celery("worker", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
# Google OAuth Configurations  # Only for testing, remove for production
CLIENT_SECRETS_FILE = "client_secret.json"  # Path to your client_secret.json file
SCOPES = ['https://www.googleapis.com/auth/drive.file']
REDIRECT_URI = 'http://localhost:8000/callback'  # Adjust based on your callback URL

def get_google_auth_flow(state=None):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI
    )
    return flow


def login():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None or gauth.access_token_expired:
        # Authenticate if they're not there or if expired
        gauth.CommandLineAuth()  # Use CommandLineAuth for manual authentication
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("credentials.json")
    drive = GoogleDrive(gauth)
    return drive

# Function to upload a file to Google Drive
def upload_file(drive, file_path, file_name, google_drive_folder_id):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False  
    # Create a file in the specified folder with the correct MIME type for a video
    file_metadata = {
        'title': file_name,
        'parents': [{'id': google_drive_folder_id}]
    }
    gfile = drive.CreateFile(file_metadata)
    gfile.SetContentFile(file_path)
    gfile.Upload()  # Upload the file

    print(f"Uploaded file with ID: {gfile['id']}")
    return True


def download_video(video_url, filename):
    full_path = f'tmp/{filename}'
    if os.path.isfile(full_path):
        print(f"The file '{filename}' already exists. Skipping download.")
        return
    
    ydl_opts = {
        'outtmpl': full_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # Ensure the final file format is mp4
        }],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            print(f"Downloaded video to {full_path}")
    except DownloadError as e:
        print(f"ERROR: Unable to download video data: {e}")
        return None

def fetch_trending_keywords():
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        trending_searches_df = pytrends.trending_searches()
        trending_terms = trending_searches_df.head(50)[0].tolist()
        return trending_terms
    except Exception as e:
        print(f"An exception occurred while fetching trending keywords: {e}")
        return []

def generate_segments(response):
    for i, segment in enumerate(response):
        print(i, segment)

        # Convert start and end times to formatted strings if they are not strings already
        start_time_seconds = segment.get("start_time", 0) if isinstance(segment.get("start_time"), float) else segment.get("start_time", "0")
        end_time_seconds = segment.get("end_time", 0) if isinstance(segment.get("end_time"), float) else segment.get("end_time", "0")
        
        # Format times as strings if they were given as floats
        start_time = format_time_from_seconds(start_time_seconds) if isinstance(start_time_seconds, float) else start_time_seconds
        end_time = format_time_from_seconds(end_time_seconds) if isinstance(end_time_seconds, float) else end_time_seconds

        # Ensuring the time format is in HH:MM:SS for FFmpeg command compatibility
        start_time_formatted = start_time.split('.')[0] if '.' in start_time else start_time
        end_time_formatted = end_time.split('.')[0] if '.' in end_time else end_time

        # Format the FFmpeg command with the formatted time strings
        output_file = f"output{str(i).zfill(3)}.mp4"
        command = f"ffmpeg -y -i tmp/input_video.mp4 -vf scale='1920:1080' -q:v 3 -b:v 6000k -ss {start_time_formatted} -to {end_time_formatted} tmp/{output_file}"
        subprocess.call(command, shell=True)

def format_time_from_seconds(seconds):
    # Helper function to format the time string from seconds
    return str(timedelta(seconds=seconds))

def generate_subtitle(input_file, output_folder, results_folder):
    command = f"auto_subtitle tmp/{input_file} -o {results_folder}/{output_folder} --model medium"
    print (command)
    subprocess.call(command, shell=True)

def generate_short(input_file, output_file):
    # Initialize face_positions as an empty list to handle the case where no faces are found.
    face_positions = []
    switch_interval = 150
    frame_count = 0
    current_face_index = -1  # Using -1 to increment it before using.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_center = None 
    cap = cv2.VideoCapture(f"tmp/{input_file}")
    if not cap.isOpened():
        print(f"Failed to open video file: tmp/{input_file}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"tmp/{output_file}", fourcc, 30, (1080, 1920))

    try:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % switch_interval == 0 or not face_positions:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))
                
                if faces:
                    face_positions = [face for face in faces]  # Update face_positions with new detections
                    current_face_index = (current_face_index + 1) % len(face_positions)
                    x, y, w, h = face_positions[current_face_index]

                    print (f"Current Face index {current_face_index} heigth {h} width {w} total faces {len(face_positions)}")

                    face_center = (x + w//2, y + h//2)

                    if w * 16 > h * 9:
                        w_916 = w
                        h_916 = int(w * 16 / 9)
                    else:
                        h_916 = h
                        w_916 = int(h * 9 / 16)

                    #Calculate the target width and height for cropping (vertical format)
                    if max(h, w) < 345:
                        target_height = int(frame_height * CROP_RATIO_SMALL)
                        target_width = int(target_height * VERTICAL_RATIO)
                    else:
                        target_height = int(frame_height * CROP_RATIO_BIG)
                        target_width = int(target_height * VERTICAL_RATIO)

                # Calculate the top-left corner of the 9:16 rectangle
                x_916 = (face_center[0] - w_916 // 2)
                y_916 = (face_center[1] - h_916 // 2)

                crop_x = max(0, x_916 + (w_916 - target_width) // 2)  # Adjust the crop region to center the face
                crop_y = max(0, y_916 + (h_916 - target_height) // 2)
                crop_x2 = min(crop_x + target_width, frame_width)
                crop_y2 = min(crop_y + target_height, frame_height)


                # Crop the frame to the face region
                crop_img = frame[crop_y:crop_y2, crop_x:crop_x2]
                
                resized = cv2.resize(crop_img, (1080, 1920), interpolation = cv2.INTER_AREA)
                
                out.write(resized)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
            # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()


            # Extract audio from original video
    audio_command = f"ffmpeg -y -i tmp/{input_file} -vn -acodec copy tmp/output-audio.aac"
    subprocess.call(audio_command, shell=True)
    combine_command = f"ffmpeg -y -i tmp/{output_file} -i tmp/output-audio.aac -c copy tmp/final-{output_file}"
    subprocess.call(combine_command, shell=True)



def generate_viral(transcript, openai_api_key, chunk_size=1000, min_duration=30, max_duration=61):
    # Break the transcript into chunks
    openai.api_key = openai_api_key
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    viral_segments = []
    json_template = '''
        { "segments" :
            [
                {
                    "start_time": 00.00, 
                    "end_time": 00.00,
                    "description": "Description of the text",
                    "duration":00,
                },    
            ]
        }
    '''

    for chunk in chunks:
        prompt = f"""
        Given the following video transcript, analyze each part for potential virality and identify 3 most viral segments from the transcript. Each segment should have nothing less than 50 seconds in duration. The provided transcript is as follows: {transcript}. Based on your analysis, return a JSON document containing the timestamps (start and end), the description of the viral part, and its duration. The JSON document should follow this format: {json_template}. Please replace the placeholder values with the actual results from your analysis. Segments should be between {min_duration} and {max_duration} seconds. Transcript: {chunk}
        """
        system = f"You are a Viral Segment Identifier, an AI system that analyzes a video's transcript and predict which segments might go viral on social media platforms. You use factors such as emotional impact, humor, unexpected content, and relevance to current trends to make your predictions. You return a structured JSON document detailing the start and end times, the description, and the duration of the potential viral segments."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            max_tokens=512,
            n=1,
            stop=None,
            timeout=900
        )

        try:
            # Correctly extracting the text from the response
            response_text = response['choices'][0]['message']['content'].strip()
            segments = json.loads(response_text)  # Parse the text as JSON
            viral_segments.extend(segments.get('segments', []))  # Append the segments
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        except AttributeError as e:
            print(f"AttributeError: {e}")

    return viral_segments

def get_video_title(video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    if 'items' in response:
        title = response['items'][0]['snippet']['title']
        return title
    else:
        return None


def generate_subtitle(input_file, output_folder, results_folder):
    command = f"auto_subtitle tmp/{input_file} -o {results_folder}/{output_folder} --model medium"
    print (command)
    subprocess.call(command, shell=True)

def generate_transcript(input_file):
    command = f"auto_subtitle tmp/{input_file} --srt_only True --output_srt True -o tmp/ --model medium"
    subprocess.call(command, shell=True)
    
    # Read the contents of the input file
    try:
        with open(f"tmp/{os.path.basename(input_file).split('.')[0]}.srt", 'r', encoding='utf-8') as file:
            transcript = file.read()
    except IOError:
        print("Error: Failed to read the input file.")
        sys.exit(1)
    return transcript


def extract_video_id(url: str) -> str:
    # Regular expression for extracting the YouTube video ID
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.match(youtube_regex, url)
    if youtube_regex_match:
        return youtube_regex_match.group(6)
    raise ValueError("Invalid YouTube URL")

def process_single_video(args):
    if not args.video_id and not args.file:
        print('Needed at least one argument. <command> --help for help')
        sys.exit(1)

    if args.video_id and args.file:
        print('use --video_id or --file')
        sys.exit(1)


# FastAPI application instance
app = FastAPI()

# Video Request Model
class VideoRequest(BaseModel):
    url: str
    openai_api_key: str
    google_drive_folder_id: str

# Celery task for processing video
@celery_app.task(name="process_video_task")
def process_video_task(video_url, google_drive_folder_id, openai_api_key):
    min_duration = 30  # Minimum duration in seconds
    max_duration = 61 
    try:
        video_id = extract_video_id(video_url)
    except ValueError:
        print("Invalid YouTube URL")
        title = get_video_title(video_id)
        video_url = 'https://www.youtube.com/watch?v=' + video_id
        print(f"Processing video with URL: {video_url}")
    # Create a temp folder
    try:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.mkdir('tmp')
    except OSError as error:
        print(error)

    filename = 'input_video.mp4'
    
    # Download the video
    download_video(video_url, filename)

    output_folder = 'results'

    try:
        os.mkdir(f"{output_folder}")
    except OSError as error:
        print(error)
    
    try:
        os.mkdir(f"{output_folder}/{video_id}")
    except OSError as error:
        print(error)

    # Verify if the output_file exists or create it
    output_file = f"{output_folder}/{video_id}/content.txt"
    
    if not os.path.exists(output_file):
        # Generate transcriptions
        transcript = generate_transcript(filename)
        print(transcript)


        viral_segments = generate_viral(transcript, openai_api_key, chunk_size=1000, min_duration=30, max_duration=61)
        if isinstance(viral_segments, list) and viral_segments:
            for segment in viral_segments:
        # Example processing, adjust according to your needs
                start_time = segment.get("start_time")
                end_time = segment.get("end_time")
                description = segment.get("description")
                
                content = json.dumps({"segments": viral_segments}, ensure_ascii=False)
                try:
                    with open(output_file, 'w', encoding='utf-8') as file:
                        file.write(content)
                except IOError:
                    print("Error: Failed to write the output file.")
                    sys.exit(1)
        else:
            print("Error: No viral segments were found or the format is incorrect.")
            return

    else:
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                content = file.read()
        except IOError:
            print("Error: Failed to read the input file.")
            sys.exit(1)

    # Check the content before parsing
    if content.strip():  # Checks if content is not empty or just whitespace
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)

        if isinstance(parsed_content, dict) and "segments" in parsed_content:
            generate_segments(parsed_content['segments'])
        else:
            print("Error: Expected JSON content with a 'segments' key.")
            sys.exit(1)

    # Loop through each segment
    for i, segment in enumerate(parsed_content['segments']):
        input_file = f'output{str(i).zfill(3)}.mp4'
        output_file = f'output_cropped{str(i).zfill(3)}.mp4'
        generate_short(input_file, output_file)


    # Logs into the drive using PyDrive authentication
    drive = login()

    # Presumed Google Drive folder ID where you want to upload files
    folder_id = google_drive_folder_id ##user should input the google drive id

    # Fetch trending keywords to process videos

    output_folder_path = f'tmp/'
    output_files = os.listdir(output_folder_path)

                # Upload each output file
    for output_file in output_files:
        if output_file.startswith('output_cropped'):
            output_file_path = f'{output_folder_path}/{output_file}'
            if os.path.exists(output_file_path):  # Check if the file exists before uploading
                upload_success = upload_file(drive, output_file_path, output_file, folder_id)
                if upload_success:
                    print(f"Uploaded {output_file} to Google Drive.")
                else:
                    print(f"Failed to upload {output_file}.")
            else:
                print(f"File does not exist and thus wasn't uploaded: {output_file_path}")
        else:
            print(f"Skipping file {output_file}, does not match upload criteria.")
    else:
        print(f"The directory {output_folder_path} does not exist or no output to upload.")
                    
app = FastAPI()


class VideoRequest(BaseModel):
    url: str
    openai_api_key: str  # User's OpenAI API Key
    google_drive_folder_id: str  # User's Google Drive Folder ID


# FastAPI Endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the Video Processing API!"}

@app.post("/process_video_task/")
async def process_video_endpoint(request: Request):
    request_data = await request.json()
    task = process_video_task.delay(
        request_data["video_url"],
        request_data["google_drive_folder_id"],
        request_data["openai_api_key"]
    )

@app.get("/authorize")
async def authorize():
    flow = get_google_auth_flow()
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    return {'authorization_url': authorization_url, 'state': state}

@app.get("/callback")
async def oauth2callback(state: str, code: str):
    flow = get_google_auth_flow(state=state)
    flow.fetch_token(code=code)
    credentials = flow.credentials
    # Save credentials securely for the user session
    # Implement user session and credentials storage logic here
    return {"detail": "Authorization successful"}


    # Your logic for processing a single video specified by args
   
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

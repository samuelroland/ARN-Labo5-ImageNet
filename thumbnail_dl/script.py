import os

import requests
import yt_dlp

# YouTube channel URLs
CHANNEL_URLS = {
    'Underscore': 'https://www.youtube.com/@Underscore_/videos',
    'Computerphile': 'https://www.youtube.com/@Computerphile/videos'
}

# Directory to save thumbnails
THUMBNAIL_DIR = 'thumbnails'

TEST_DIR = 'test'

# Create directory if it doesn't exist
if not os.path.exists(THUMBNAIL_DIR):
    os.makedirs(THUMBNAIL_DIR)

if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

def download_thumbnails(channel_name, channel_url, max_results=250):
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'playlistend': max_results
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)

        video_entries = result.get('entries', [])
        video_ids = [entry['id'] for entry in video_entries]

        channel_dir = os.path.join(THUMBNAIL_DIR, channel_name.replace(' ', '_'))
        if not os.path.exists(channel_dir):
            os.makedirs(channel_dir)


        for i, video_id in enumerate(video_ids):
            if i == 200:
                channel_dir = os.path.join(TEST_DIR, channel_name.replace(' ', '_'))
                if not os.path.exists(channel_dir):
                    os.makedirs(channel_dir)
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                with open(os.path.join(channel_dir, f'thumbnail_{i+1}.jpg'), 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download thumbnail {i+1} from {channel_name}")

for channel_name, channel_url in CHANNEL_URLS.items():
    print(f"Fetching thumbnails for {channel_name}...")
    download_thumbnails(channel_name, channel_url)
    print(f"Downloaded thumbnails for {channel_name}")

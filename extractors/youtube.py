import re
import logging
from typing import Union, List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> Optional[str]: 
    """
    Extracts the YouTube video ID from a URL.
    Supports various YouTube URL formats.

    Args:
        url: The YouTube video URL.

    Returns:
        The video ID as a string, or None if not found.
    """
    # Regex patterns to match YouTube video IDs
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',  # Standard watch URL
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',          # Shortened URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})', # Embed URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',      # v/ URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})' # Shorts URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

def validate_youtube_url(url: str) -> bool:
    """
    Validates if the given URL is a recognizable YouTube video URL
    by trying to extract a video ID.

    Args:
        url: The URL string to validate.

    Returns:
        True if a video ID can be extracted, False otherwise.
    """
    return extract_video_id(url) is not None

def get_video_metadata(url: str) -> Optional[Dict]: 
    """
    Extracts video metadata using yt-dlp.

    Args:
        url: The YouTube video URL.

    Returns:
        A dictionary containing metadata (title, channel, duration, etc.)
        or None if extraction fails.
    """
    ydl_opts = {
        'quiet': True,        # Suppress console output
        'no_warnings': True,  # Suppress warnings
        'skip_download': True, # Don't download the video
        'force_generic_extractor': False # Use youtube extractor
    }
    try:
        # Use yt-dlp context manager for resource management
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract information without downloading the video file
            info_dict = ydl.extract_info(url, download=False)
            # Construct metadata dictionary from extracted info
            metadata = {
                'video_id': info_dict.get('id'),
                'title': info_dict.get('title'),
                'channel': info_dict.get('uploader'),
                'channel_url': info_dict.get('uploader_url'),
                'upload_date': info_dict.get('upload_date'), # Format YYYYMMDD
                'duration': info_dict.get('duration'), # In seconds
                'description': info_dict.get('description'),
                'thumbnail': info_dict.get('thumbnail'),
                'view_count': info_dict.get('view_count'),
                'like_count': info_dict.get('like_count'),
                'tags': info_dict.get('tags'),
                'url': url # Include original URL for reference
            }
            logger.info(f"Successfully extracted metadata for {url}")
            return metadata
    except yt_dlp.utils.DownloadError as e:
        # Handle errors specifically related to downloading/extracting info
        logger.error(f"yt-dlp failed to extract metadata for {url}: {e}")
        # Provide more specific feedback based on error message content
        if "Private video" in str(e):
            logger.error("Reason: Video is private.")
        elif "Video unavailable" in str(e):
            logger.error("Reason: Video is unavailable.")
        elif "Login required" in str(e):
             logger.error("Reason: Login required (possibly private or members-only).")
    except Exception as e:
        # Catch any other unexpected exceptions during the process
        logger.error(f"An unexpected error occurred during metadata extraction for {url}: {e}")
    # Return None if any exception occurred or metadata couldn't be retrieved
    return None

def get_transcript(video_id: str, languages: List[str] = ['en']) -> Optional[str]: 
    """
    Fetches the transcript for a given YouTube video ID using youtube_transcript_api.

    Args:
        video_id: The 11-character YouTube video ID.
        languages: A list of preferred language codes in order of preference
                   (e.g., ['en', 'es']). Defaults to English ['en'].

    Returns:
        The transcript text as a single string, or None if unavailable or an error occurs.
    """
    try:
        # Retrieve the list of available transcripts for the video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Attempt to find a manually created transcript in the specified languages
        # find_transcript raises NoTranscriptFound if no suitable transcript is found
        transcript = transcript_list.find_transcript(languages)
        logger.info(f"Found manually created transcript in language: {transcript.language} for video ID: {video_id}")

        # Fetch the actual transcript data (list of snippet objects)
        transcript_data = transcript.fetch()

        # *** FIX: Use attribute access (item.text) instead of dictionary access (item['text']) ***
        full_transcript = " ".join([item.text for item in transcript_data])
        logger.info(f"Successfully fetched transcript for video ID: {video_id}")
        return full_transcript

    except TranscriptsDisabled:
        # Handle cases where transcripts are explicitly disabled for the video
        logger.warning(f"Transcripts are disabled for video ID: {video_id}")
        # Fallback idea: Implement yt-dlp subtitle download here if desired.
        return None
    except NoTranscriptFound:
        # Handle cases where no manual transcript exists in the desired languages
        logger.warning(f"No manual transcript found for video ID: {video_id} in languages: {languages}. Trying auto-generated...")
        # As a fallback, try finding an auto-generated (ASR) transcript
        try:
            # find_generated_transcript raises NoTranscriptFound if none exist in the languages
            generated_transcript = transcript_list.find_generated_transcript(languages)
            logger.info(f"Found auto-generated transcript in language: {generated_transcript.language} for video ID: {video_id}")
            transcript_data = generated_transcript.fetch()

            # *** FIX: Use attribute access (item.text) instead of dictionary access (item['text']) ***
            full_transcript = " ".join([item.text for item in transcript_data])
            logger.info(f"Successfully fetched auto-generated transcript for video ID: {video_id}")
            return full_transcript
        except NoTranscriptFound:
             # Handle cases where no auto-generated transcript is found either
             logger.warning(f"No auto-generated transcript found either for video ID: {video_id} in languages: {languages}")
             return None
        except Exception as e_gen:
            # Catch errors specific to fetching the generated transcript
            logger.error(f"An error occurred fetching the auto-generated transcript for video ID {video_id}: {e_gen}")
            return None
    except Exception as e:
        # Catch any other unexpected errors from the youtube_transcript_api
        # Including potential issues if transcript_data structure changes unexpectedly
        logger.error(f"An unexpected error occurred fetching transcript for video ID {video_id}: {e}")
        return None

def save_transcript_to_markdown(metadata: Dict, transcript: Optional[str], output_dir: str = "transcripts") -> None:
    """
    Saves the transcript to a markdown file with metadata.
    
    Args:
        metadata: Dictionary containing video metadata
        transcript: The transcript text or None
        output_dir: Directory to save the markdown file
    """
    if not transcript:
        logger.warning("No transcript to save")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize title for filename
    safe_title = "".join(c for c in metadata['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"{output_dir}/{safe_title}_{metadata['video_id']}.md"
    
    # Create markdown content
    markdown_content = f"""# {metadata['title']}

    ## Video Metadata
    - **Video ID**: {metadata['video_id']}
    - **Channel**: {metadata['channel']}
    - **Upload Date**: {metadata['upload_date']}
    - **Duration**: {metadata['duration']} seconds
    - **URL**: {metadata['url']}

    ## Transcript
    {transcript}
    """
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Successfully saved transcript to {filename}")
    except Exception as e:
        logger.error(f"Failed to save transcript to {filename}: {e}")

def extract_content(url: str, transcript_languages: List[str] = ['en']) -> Optional[Dict]: 
    """
    Main function to validate a YouTube URL, extract metadata, and fetch the transcript.

    Args:
        url: The YouTube video URL.
        transcript_languages: A list of preferred language codes for the transcript,
                              in order of preference. Defaults to ['en'].

    Returns:
        A dictionary containing 'metadata' (dict) and 'transcript' (str or None),
        or None if the URL is invalid or metadata extraction fails critically.
    """
    logger.info(f"Starting content extraction for URL: {url}")

    # 1. Validate URL and extract Video ID
    video_id = extract_video_id(url)
    if not video_id:
        logger.error(f"Invalid or unsupported YouTube URL: {url}")
        return None 

    # 2. Extract Metadata
    metadata = get_video_metadata(url)
    if not metadata:
        logger.error(f"Failed to extract metadata for video ID: {video_id} from URL: {url}. Aborting extraction.")
        return None 

    # 3. Extract Transcript (using the extracted video_id)
    transcript = get_transcript(video_id, languages=transcript_languages)
    if not transcript:
        logger.warning(f"Could not retrieve transcript for video ID: {video_id}. Proceeding without it.")
        
    save_transcript_to_markdown(metadata, transcript)

    # 4. Return combined results
    logger.info(f"Content extraction completed for video ID: {video_id}")
    return {
        "metadata": metadata, # Dictionary of metadata
        "transcript": transcript # String or None
    }

if __name__ == '__main__':
    
    test_url_valid = "https://www.youtube.com/watch?v=v34Eg12mhDM"

    print("\n--- Testing Valid URL ---")
    content = extract_content(test_url_valid)
    if content:
        print("Metadata:", content['metadata']['title'])
        print("Transcript available:", bool(content['transcript']))
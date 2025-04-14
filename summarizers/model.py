import logging
import requests
import time
import json
import re
import os
from typing import Optional, List, Dict, Any

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
MINIMAX_API_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-Text-01" # Or other compatible model

# Caching mechanism (simple in-memory dictionary)
summary_cache: Dict[str, str] = {}

# Configuration loading
CONFIG_FILE = 'config.json'
MINIMAX_API_KEY = None # Will be loaded by load_minimax_config

def load_minimax_config() -> Optional[str]:
    """Loads the MiniMax API key from config.json or environment variable."""
    global MINIMAX_API_KEY
    if MINIMAX_API_KEY: return MINIMAX_API_KEY
    MINIMAX_API_KEY = None
    try:
        # Determine the absolute path to the config file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of the current script
        config_path = os.path.normpath(os.path.join(script_dir, '..', CONFIG_FILE)) # Go up one level for config.json

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                loaded_key = config.get("minimax_api_key")
                if loaded_key and loaded_key != "YOUR_MINIMAX_API_KEY_HERE":
                    MINIMAX_API_KEY = loaded_key
                    logger.info("Loaded MiniMax API key from config.json.")
                    return MINIMAX_API_KEY
                else: logger.warning("MiniMax API key not found or not set in config.json.")
        else: logger.warning(f"Configuration file not found at: {config_path}")
    except Exception as e: logger.error(f"An error occurred loading configuration from file: {e}")

    logger.info("Attempting to load MiniMax API key from environment variable MINIMAX_API_KEY.")
    env_key = os.environ.get("MINIMAX_API_KEY")
    if env_key:
        MINIMAX_API_KEY = env_key
        logger.info("Loaded MiniMax API key from environment variable MINIMAX_API_KEY.")
        return MINIMAX_API_KEY
    else:
        logger.error("MiniMax API key is missing. Please add it to config.json or set MINIMAX_API_KEY environment variable.")
        return None

# --- Helper Functions ---

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: normalize whitespace."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- API Interaction ---

def query_minimax_api(
    prompt: str,
    api_key: str,
    system_prompt: Optional[str] = "You are a helpful assistant that summarizes text accurately and concisely.",
    tokens_to_generate: int = 300, # Default token limit for the summary
    retries: int = 3,
    delay: int = 5
) -> Optional[str]: # Returns the generated text content or None
    """
    Sends a request to the MiniMax Chat Completion v2 API.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = []
    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Add the main user prompt
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MINIMAX_MODEL,
        "messages": messages,
        "tokens_to_generate": tokens_to_generate,
        "temperature": 0.7, # Controls randomness, lower is more deterministic
        "top_p": 0.95 # Nucleus sampling parameter
    }
    logger.debug(f"Querying MiniMax API: {MINIMAX_API_URL}")
    logger.debug(f"Payload keys: {payload.keys()}, Prompt length (chars): {len(prompt)}") # Log prompt length

    for attempt in range(retries + 1):
        try:
            # Send the POST request
            response = requests.post(MINIMAX_API_URL, headers=headers, json=payload, timeout=120) # Increased timeout
            response_json = None
            try:
                # Attempt to parse the JSON response
                response_json = response.json()
                # Check for MiniMax specific error structure in the response body
                if response_json and 'base_resp' in response_json:
                    base_resp = response_json['base_resp']
                    if base_resp.get('status_code') != 0:
                        logger.error(f"MiniMax API Error: Code {base_resp.get('status_code')}, Msg: {base_resp.get('status_msg')}")
                        # Add specific error handling based on MiniMax documentation if needed
                        # Example: Check for input length error codes
                        # if base_resp.get('status_code') == ERROR_CODE_FOR_TOO_LONG:
                        #    logger.error("Input transcript likely exceeded API limits.")
                        return None # Fail on MiniMax specific errors
            except json.JSONDecodeError:
                 # Log if response is not valid JSON, but continue to check HTTP status
                 logger.warning(f"Could not decode JSON response. Status: {response.status_code}, Body: {response.text[:200]}...")
                 pass # Allow checking HTTP status code even if JSON parsing failed

            # Check HTTP status codes
            if response.status_code == 200 and response_json:
                # Check the expected structure of a successful response
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    message = response_json['choices'][0].get('message', {})
                    content = message.get('content')
                    if content:
                        return content.strip() # Return the generated content
                    else:
                        # Log error if content is missing in a successful response
                        logger.error("MiniMax API response missing 'content' in choices message.")
                        return None
                else:
                    # Log error if choices array is missing or empty
                    logger.error("MiniMax API response missing 'choices' or choices list is empty.")
                    return None
            elif response.status_code == 400: # Bad Request (often input too long)
                 logger.error(f"MiniMax API returned HTTP 400 (Bad Request). Input may be too long or malformed. Response: {response.text[:200]}...")
                 return None # Usually not retryable
            elif response.status_code == 401: # Unauthorized
                 logger.error(f"MiniMax API returned HTTP 401 (Unauthorized). Check API Key. Response: {response.text[:200]}...")
                 return None # Not retryable
            elif response.status_code == 429: # Rate Limit
                 logger.error(f"MiniMax API returned HTTP 429 (Rate Limit Exceeded). Response: {response.text[:200]}...")
                 # Optionally add a longer delay or stop retrying immediately for rate limits
                 if attempt < retries: time.sleep(delay * 2) # Wait longer for rate limits
                 else: logger.error("Max retries reached after rate limit."); return None
            elif response.status_code >= 500: # Server errors (retryable)
                 logger.warning(f"MiniMax API returned HTTP {response.status_code}. Retrying... (Attempt {attempt + 1}/{retries + 1})")
                 if attempt < retries: time.sleep(delay)
                 else: logger.error(f"Max retries reached for HTTP {response.status_code}."); return None
            else: # Handle other unexpected HTTP status codes
                logger.error(f"MiniMax API request failed with unexpected HTTP status {response.status_code}: {response.text[:200]}...")
                return None # Not retryable

        except requests.exceptions.Timeout:
             # Handle request timeout
             logger.error(f"API request to {MINIMAX_API_URL} timed out after 120 seconds.")
             if attempt < retries: logger.warning(f"Retrying timeout..."); time.sleep(delay)
             else: logger.error("Max retries reached for timeout."); return None
        except requests.exceptions.RequestException as e:
            # Handle network-related errors
            logger.error(f"Network error during API request: {e}")
            if attempt < retries: logger.warning(f"Retrying network error..."); time.sleep(delay)
            else: logger.error("Max retries reached for network error."); return None
        except Exception as e:
             # Handle any other unexpected errors during the process
             logger.error(f"An unexpected error occurred during MiniMax API query: {e}", exc_info=True)
             return None # Don't retry on unexpected errors
    # Return None if all retries fail
    return None

# --- File Saving Function ---
def save_summary_to_markdown(
    video_id: str,
    summary: str,
    title: Optional[str] = None,
    output_dir: str = "summaries"
) -> Optional[str]:
    """Saves a video summary to a markdown file."""
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Sanitize video_id for use in filename
        safe_video_id = re.sub(r'[^\w\-]+', '_', video_id) # Replace non-alphanumeric chars (except hyphen) with underscore
        filename = f"{safe_video_id}.md"
        file_path = os.path.join(output_dir, filename)
        # Get current date for metadata
        current_date = time.strftime("%Y-%m-%d")
        # Prepare markdown content with YAML frontmatter
        markdown_content = (
            f"---\n"
            f"video_id: {video_id}\n"
            f"title: {title or video_id}\n" # Use video_id as title if none provided
            f"date: {current_date}\n"
            f"---\n\n"
            f"# Summary: {title or f'Video {video_id}'}\n\n"
            f"{summary}\n" # Add the summary content
        )
        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Successfully saved summary to {file_path}")
        return file_path # Return the path to the saved file
    except Exception as e:
        # Log any errors during file saving
        logger.error(f"Failed to save summary to markdown file '{file_path}': {e}")
        return None # Indicate failure

# --- Main Summarization Function (Single Call) ---

def summarize_transcript(
    transcript: str,
    video_id: str,
    title: Optional[str] = None,
    # Target token count for the single summary
    summary_tokens: int = 300, # Target length for the final summary
    save_to_file: bool = True,
    output_dir: str = "summaries_minimax_single" # Use different dir
) -> Optional[str]:
    """
    Summarizes the entire transcript using a single MiniMax API call.

    Args:
        transcript: The full transcript text.
        video_id: Unique video identifier (for caching).
        title: Optional video title for saving.
        summary_tokens: Target token length for the final summary.
        save_to_file: Whether to save the result to a markdown file.
        output_dir: Directory to save markdown files.

    Returns:
        The final summary string, or None on failure.

    Warning: This may fail if the transcript exceeds the API's input limits.
    """
    # 1. Load API Key
    api_key = load_minimax_config()
    if not api_key:
        logger.error("Cannot summarize: Missing MiniMax API Key.")
        return None

    # 2. Check cache
    cache_key = f"{video_id}_minimax_single" # Use specific key
    if cache_key in summary_cache:
        logger.info(f"Returning cached summary for key: {cache_key}")
        return summary_cache[cache_key]

    logger.info(f"Starting MiniMax single-call summarization for video ID: {video_id}")

    # 3. Preprocess
    processed_text = preprocess_text(transcript)
    if not processed_text:
        logger.warning("Transcript is empty after preprocessing.")
        return ""

    # 4. Prepare prompt for single API call using the user's custom prompt
    # Incorporate the user's specific instructions
    prompt = (
        f"Summarize the following YouTube video transcript in a clear and concise way, "
        f"highlighting the main points, key arguments, and important takeaways. "
        f"Use bullet points or short paragraphs for clarity. Do not include filler or redundant content.\n\n"
        f"Transcript:\n---\n{processed_text}\n---\n\nSummary:"
    )

    # 5. Call MiniMax API once
    logger.info("Sending entire transcript to MiniMax API for summarization...")
    final_summary = query_minimax_api(
        prompt=prompt,
        api_key=api_key,
        tokens_to_generate=summary_tokens
        # System prompt is default, can be overridden if needed:
        # system_prompt="Custom system message if needed"
    )

    if not final_summary:
        logger.error("Failed to generate summary from MiniMax API.")
        return None # Summarization failed

    logger.info("Successfully generated summary via single MiniMax API call.")

    # 6. Update cache
    logger.info(f"Caching final summary under key: {cache_key}")
    summary_cache[cache_key] = final_summary

    # 7. Save to markdown file if requested
    if save_to_file:
        file_path = save_summary_to_markdown(
            video_id=video_id,
            summary=final_summary,
            title=title,
            output_dir=output_dir
        )
        # Logging about success/failure happens within save_summary_to_markdown

    return final_summary


# ---------- #
# -- MAIN -- #
# ---------- #

if __name__ == '__main__':
    # Ensure config is loaded for testing
    loaded_key = load_minimax_config()
    if not loaded_key:
         print("CRITICAL: MiniMax API Key could not be loaded. Check config.json or MINIMAX_API_KEY env var. Exiting.")
         exit()
    # Mask key for printing only (shows first 5 and last 4 chars)
    masked_key = loaded_key[:5] + "****" + loaded_key[-4:] if len(loaded_key) > 9 else loaded_key[:3] + "****"
    print(f"Using MiniMax API Key: {masked_key}")

    # --- Test Cases ---
    # Short example transcript:
    example_transcript_short = """
    The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy.
    As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old,
    distant, or faint for the Hubble Space Telescope. It launched in December 2021.
    """
    # Longer example transcript (increase multiplier to test potential API limits):
    example_transcript_long = """
    The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy.
    As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old,
    distant, or faint for the Hubble Space Telescope. This enables investigations across many fields of astronomy
    and cosmology, such as observation of the first stars and the formation of the first galaxies, and detailed
    atmospheric characterization of potentially habitable exoplanets. The U.S. National Aeronautics and Space
    Administration (NASA) led JWST's development in collaboration with the European Space Agency (ESA) and the
    Canadian Space Agency (CSA). The telescope is named after James E. Webb, who was the administrator of NASA
    from 1961 to 1968 during the Mercury, Gemini, and Apollo programs. The JWST was launched on 25 December 2021
    on an ESA Ariane 5 rocket from Kourou, French Guiana, and arrived at the Sun–Earth L2 Lagrange point in January 2022.
    The first JWST image was released to the public via a press conference on 11 July 2022. The telescope's primary
    mirror consists of 18 hexagonal mirror segments made of gold-plated beryllium which combine to create a 6.5-meter
    (21 ft) diameter mirror – considerably larger than Hubble's 2.4 m (7.9 ft) mirror. Unlike Hubble, which observes
    in the near ultraviolet, visible, and near infrared (0.1–1.7 μm) spectra, JWST observes in a lower frequency range,
    from long-wavelength visible light (red) through mid-infrared (0.6–28.3 μm). This allows it to observe high redshift
    objects that are too old and too distant for Hubble. The telescope must be kept extremely cold to observe faint signals
    in the infrared without interference from other sources of warmth. It is deployed in a solar orbit near the Sun–Earth
    L2 Lagrange point, about 1.5 million kilometers (930,000 mi) from Earth, where its five-layer, kite-shaped sunshield
    protects it from warming by the Sun, Earth, and Moon. It provides unprecedented views of the universe.
    """ * 3 # Adjust multiplier to test length

    example_video_id = "jwst_minimax_custom_prompt"
    example_title = "JWST Custom Prompt Summary"

    print(f"\n--- Testing MiniMax API Single Call (Using LONG transcript, {len(example_transcript_long)} chars) ---")
    summary_long = summarize_transcript(
        transcript=example_transcript_long,
        video_id=example_video_id,
        title=example_title,
        summary_tokens=400, # Request a slightly longer summary
        save_to_file=True,
        output_dir="summaries_minimax_single"
    )

    if summary_long:
        print("Final Summary Length (chars):", len(summary_long))
        print("\nGenerated Summary (Long Transcript):")
        print(summary_long)
    else:
        print("Summarization failed for the long transcript. It might have exceeded API limits or another error occurred.")

    # Test caching (using the same ID)
    print("\n--- Testing Caching (MiniMax Single Call) ---")
    summary_cached = summarize_transcript(example_transcript_long, example_video_id, title=example_title)
    if summary_cached:
        print("Retrieved MiniMax cached summary successfully.")
        # assert summary_long == summary_cached # Verify content if first attempt succeeded
    else:
        print("Cache test failed (likely because the initial summarization attempt failed).")

    # Optional: Test with the short transcript
    print(f"\n--- Testing MiniMax API Single Call (Using SHORT transcript, {len(example_transcript_short)} chars) ---")
    example_video_id_short = "jwst_minimax_short_prompt"
    summary_short = summarize_transcript(
        transcript=example_transcript_short,
        video_id=example_video_id_short,
        title="JWST Short Summary",
        summary_tokens=100,
        save_to_file=True,
        output_dir="summaries_minimax_single"
    )
    if summary_short:
        print("Final Summary Length (chars):", len(summary_short))
        print("\nGenerated Summary (Short Transcript):")
        print(summary_short)
    else:
        print("Summarization failed for the short transcript (check logs for errors).")
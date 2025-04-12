import logging
import re
from transformers import pipeline, AutoTokenizer
from typing import Optional, List, Dict

# Basic logging setup (integrate with logger_config.py later)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
# Choose the summarization model
# Alternatives: 'google/pegasus-xsum', 't5-small', 't5-base'
MODEL_NAME = "facebook/bart-large-cnn"
# Caching mechanism (simple in-memory dictionary)
# Key: video_id, Value: summary string
summary_cache: Dict[str, str] = {}

# --- Initialization ---
try:
    # Initialize the summarization pipeline and tokenizer
    # Using device=0 assumes CUDA GPU is available, change to device=-1 for CPU
    # Explicitly loading tokenizer to get max length info easily
    summarizer = pipeline("summarization", model=MODEL_NAME, device=-1) # Use device=0 for GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Get the model's maximum input length (e.g., 1024 for BART)
    # Subtracting some buffer for special tokens added by the tokenizer
    MODEL_MAX_LENGTH = tokenizer.model_max_length - 2 # Small buffer
    logger.info(f"Summarization pipeline initialized with model: {MODEL_NAME}")
    logger.info(f"Model max input length (tokens): {MODEL_MAX_LENGTH}")
except Exception as e:
    logger.error(f"Failed to initialize summarization pipeline: {e}")
    summarizer = None
    tokenizer = None
    MODEL_MAX_LENGTH = 512 # Default fallback if model loading fails

# --- Helper Functions ---

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing: normalize whitespace.
    Add more steps if needed (e.g., remove timestamps if present).
    """
    text = re.sub(r'\s+', ' ', text).strip()
    # Add any other specific cleaning steps here
    return text

def chunk_text(text: str, max_chunk_length_tokens: int) -> List[str]:
    """
    Chunks the text into smaller pieces based on token count,
    trying to respect sentence boundaries crudely.

    Args:
        text: The input text string.
        max_chunk_length_tokens: The maximum number of tokens allowed per chunk.

    Returns:
        A list of text chunks.
    """
    if not tokenizer:
        logger.error("Tokenizer not available for chunking.")
        # Fallback: Crude split by character count if tokenizer failed
        char_limit = max_chunk_length_tokens * 4 # Rough estimate
        return [text[i:i+char_limit] for i in range(0, len(text), char_limit)]

    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    current_chunk_start = 0

    while current_chunk_start < len(tokens):
        # Find the end position for the current chunk
        current_chunk_end = min(current_chunk_start + max_chunk_length_tokens, len(tokens))

        # Decode the chunk tokens back to text
        chunk_tokens = tokens[current_chunk_start:current_chunk_end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        # Attempt to find the last sentence boundary within the chunk for cleaner breaks
        # This is a simple approach; more robust sentence splitting could be used (e.g., nltk)
        sentence_endings = ['.', '!', '?']
        last_sentence_end_index = -1
        for ending in sentence_endings:
            index = chunk_text.rfind(ending)
            if index > last_sentence_end_index:
                last_sentence_end_index = index

        # If a sentence end is found and it's not too early in the chunk, adjust the end
        if last_sentence_end_index > len(chunk_text) * 0.7: # Ensure we don't cut too short
             # Re-tokenize the adjusted chunk to get the actual token count
             adjusted_chunk_text = chunk_text[:last_sentence_end_index + 1]
             adjusted_tokens = tokenizer.encode(adjusted_chunk_text, add_special_tokens=False)
             current_chunk_end = current_chunk_start + len(adjusted_tokens)
             chunks.append(adjusted_chunk_text)
        else:
            # If no suitable sentence end found, use the original token limit
            chunks.append(chunk_text)

        # Move to the start of the next chunk
        current_chunk_start = current_chunk_end

    logger.info(f"Chunked text into {len(chunks)} chunks.")
    return chunks


# --- Main Summarization Function ---

def summarize_transcript(
    transcript: str,
    video_id: str, # Used for caching
    max_summary_length: int = 150,
    min_summary_length: int = 30
) -> Optional[str]:
    """
    Summarizes the given transcript text.

    Args:
        transcript: The transcript text to summarize.
        video_id: The unique ID of the video (for caching).
        max_summary_length: The maximum length of the generated summary (in tokens).
        min_summary_length: The minimum length of the generated summary (in tokens).

    Returns:
        The summarized text as a string, or None if summarization fails.
    """
    if not summarizer or not tokenizer:
        logger.error("Summarizer or tokenizer not initialized. Cannot summarize.")
        return None

    # 1. Check cache
    if video_id in summary_cache:
        logger.info(f"Returning cached summary for video ID: {video_id}")
        return summary_cache[video_id]

    logger.info(f"Starting summarization for video ID: {video_id}")

    # 2. Preprocess
    processed_text = preprocess_text(transcript)
    if not processed_text:
        logger.warning("Transcript is empty after preprocessing.")
        return "" # Return empty string for empty input

    # 3. Check length and chunk if necessary
    # Tokenize to check length against model limit
    input_tokens = tokenizer.encode(processed_text, add_special_tokens=False)
    input_length = len(input_tokens)
    logger.info(f"Input transcript length (tokens): {input_length}")

    if input_length > MODEL_MAX_LENGTH:
        logger.info("Transcript exceeds model max length, chunking...")
        text_chunks = chunk_text(processed_text, MODEL_MAX_LENGTH)
    else:
        text_chunks = [processed_text] # Treat as a single chunk

    # 4. Summarize chunks
    chunk_summaries = []
    try:
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(text_chunks)}")
            # Adjust max/min length for chunks if needed, but often using the final target works okay
            # Ensure chunk isn't empty
            if not chunk.strip():
                logger.warning(f"Skipping empty chunk {i+1}")
                continue

            chunk_max_length = max_summary_length // len(text_chunks) if len(text_chunks) > 1 else max_summary_length
            chunk_min_length = min_summary_length // len(text_chunks) if len(text_chunks) > 1 else min_summary_length
            chunk_max_length = max(chunk_min_length, chunk_max_length) # Ensure max >= min
            chunk_min_length = max(10, chunk_min_length) # Ensure min length is reasonable


            # Perform summarization on the chunk
            summary_result = summarizer(
                chunk,
                max_length=chunk_max_length,
                min_length=chunk_min_length,
                do_sample=False # Use deterministic approach
            )
            chunk_summaries.append(summary_result[0]['summary_text'])

        # 5. Combine chunk summaries
        final_summary = " ".join(chunk_summaries)
        logger.info(f"Successfully generated summary for video ID: {video_id}")

        # Optional: Post-processing/Refinement of the combined summary
        # Could involve another summarization pass if the combined summary is long,
        # or sentence boundary checks, etc.
        # final_summary = post_process_summary(final_summary)

        # 6. Update cache
        summary_cache[video_id] = final_summary
        return final_summary

    except Exception as e:
        logger.error(f"An error occurred during summarization for video ID {video_id}: {e}", exc_info=True)
        return None # Indicate failure

# --- Example Usage ---
if __name__ == '__main__':
    # Example transcript (replace with actual long transcript for testing chunking)
    example_transcript = """
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
    protects it from warming by the Sun, Earth, and Moon.
    """ * 2

    example_video_id = "jwst_example"

    print("\n--- Testing Summarization ---")
    summary = summarize_transcript(example_transcript, example_video_id, max_summary_length=200, min_summary_length=50)

    if summary:
        print("\nOriginal Length (chars):", len(example_transcript))
        print("Summary Length (chars):", len(summary))
        print("\nGenerated Summary:")
        print(summary)
    else:
        print("Summarization failed.")

    # Test caching
    print("\n--- Testing Caching ---")
    summary_cached = summarize_transcript(example_transcript, example_video_id)
    if summary_cached:
        print("Retrieved cached summary successfully.")
    else:
        print("Cache test failed.")

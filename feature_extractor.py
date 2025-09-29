# feature_extractor.py
import numpy as np
import spacy
import hashlib

# Load the spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

# Define the fixed size for each feature vector to ensure consistency
TEXT_VECTOR_SIZE = nlp.vocab.vectors_length
IMAGE_VECTOR_SIZE = 16
AUDIO_VECTOR_SIZE = 4
TOTAL_VECTOR_SIZE = TEXT_VECTOR_SIZE + IMAGE_VECTOR_SIZE + AUDIO_VECTOR_SIZE

def extract_text_features(text: str | None) -> np.ndarray:
    """Processes text using spaCy to get a semantic vector."""
    if not text:
        return np.zeros(TEXT_VECTOR_SIZE)
    return nlp(text).vector

def extract_image_features(image_hash: str | None) -> np.ndarray:
    """
    SIMULATED: Processes an image hash into a feature vector.
    In a real system, you'd use a model like CLIP or a perceptual hashing library.
    """
    if not image_hash:
        return np.zeros(IMAGE_VECTOR_SIZE)
    # Create a deterministic numerical vector from the hash string
    seed = int(hashlib.md5(image_hash.encode()).hexdigest(), 16) % (10**8)
    rng = np.random.RandomState(seed)
    return rng.rand(IMAGE_VECTOR_SIZE)

def extract_audio_features(audio_info: dict | None) -> np.ndarray:
    """
    SIMULATED: Processes audio metadata into a feature vector.
    In a real system, you'd use a library like Librosa to analyze the waveform.
    """
    if not audio_info:
        return np.zeros(AUDIO_VECTOR_SIZE)
    # Use normalized features from the audio info
    duration = audio_info.get('duration_seconds', 0) / 60.0 # Normalize by 1 minute
    amplitude = audio_info.get('peak_amplitude', 0)
    clarity = audio_info.get('clarity_score', 0)
    speech_ratio = audio_info.get('speech_to_music_ratio', 0)
    return np.array([duration, amplitude, clarity, speech_ratio])


def create_feature_vector(content_data: dict) -> np.ndarray:
    """
    Creates a unified feature vector from different content types.
    """
    text_vec = extract_text_features(content_data.get('text'))
    image_vec = extract_image_features(content_data.get('image_hash'))
    audio_vec = extract_audio_features(content_data.get('audio_info'))

    # Concatenate all feature vectors into a single state vector
    return np.concatenate([text_vec, image_vec, audio_vec])
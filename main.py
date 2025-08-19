from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
import time
import uuid
import base64
from io import BytesIO
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
import torch
import re

# Pydantic models for request/response
class AssessmentRequest(BaseModel):
    level: int = 1
    set_index: int = 0
    assessment_type: str = "mixed"  # mixed, sight, phonetic
    teaching_focus: Optional[str] = None
    words: Optional[List[str]] = None

class TranscriptionRequest(BaseModel):
    audio_base64: str  # Base64 encoded audio
    expected_words: List[str]

class AssessmentResponse(BaseModel):
    accuracy_percentage: float
    words_correct: int
    total_words: int
    error_type: str
    transcription: str
    expected_words: List[str]
    word_results: List[Dict]
    current_level: int
    current_set: int
    assessment_date: str

class NextAssessmentResponse(BaseModel):
    next_level: int
    next_set: int
    next_words: List[str]
    teaching_focus: str
    focus_distribution: str
    total_words: int

class MockTranscriptionRequest(BaseModel):
    transcription: str
    expected_words: List[str]

class ActivityPresenter:
    def __init__(self):
        self.word_emojis = {
            "apple": "🍎", "banana": "🍌", "cherry": "🍒", "grape": "🍇", "orange": "🍊", "pear": "🍐",
            "dog": "🐕", "cat": "🐱", "mouse": "🐭", "lion": "🦁", "tiger": "🐅", "bear": "🐻",
            "car": "🚗", "bike": "🚲", "train": "🚆", "plane": "✈️", "boat": "⛵", "bus": "🚌",
            "house": "🏠", "tree": "🌳", "flower": "🌸", "bird": "🐦", "fish": "🐟", "star": "⭐",
            "book": "📚", "school": "🏫", "friend": "👥", "family": "👨‍👩‍👧‍👦", "happy": "😊", "play": "🎮"
        }
        
        self.phonetic_map = {
            "apple": "AP-PLE", "banana": "BA-NA-NA", "cherry": "CHER-RY",
            "grape": "GRAPE", "orange": "OR-ANGE", "pear": "PEAR",
            "dog": "DOG", "cat": "CAT", "mouse": "MOUSE",
            "lion": "LI-ON", "tiger": "TI-GER", "bear": "BEAR",
            "car": "CAR", "bike": "BIKE", "train": "TRAIN",
            "plane": "PLANE", "boat": "BOAT", "bus": "BUS",
            "house": "HOUSE", "tree": "TREE", "flower": "FLOW-ER",
            "bird": "BIRD", "fish": "FISH", "star": "STAR",
            "book": "BOOK", "school": "SCHOOL", "friend": "FRIEND",
            "family": "FAM-I-LY", "happy": "HAP-PY", "play": "PLAY"
        }
    
    def get_phonetic_breakdown(self, word):
        return self.phonetic_map.get(word, word.upper())
    
    def get_sight_display(self, word):
        emoji = self.word_emojis.get(word, "❓")
        return f"{word} → {emoji}"

class DecisionEngine:
    def __init__(self):
        self.fry_words = {
            1: [
                ["apple", "cherry", "orange", "banana", "grape", "pear"],
                ["banana", "grape", "pear", "apple", "cherry", "orange"],
                ["cherry", "orange", "apple", "banana", "grape", "pear"],
            ],
            2: [
                ["dog", "mouse", "tiger", "cat", "lion", "bear"],
                ["cat", "lion", "bear", "dog", "mouse", "tiger"],
                ["mouse", "tiger", "dog", "cat", "lion", "bear"],
            ],
            3: [
                ["car", "train", "boat", "bike", "plane", "bus"],
                ["bike", "plane", "bus", "car", "train", "boat"],
                ["train", "boat", "car", "bike", "plane", "bus"],
            ],
            4: [
                ["house", "tree", "flower", "bird", "fish", "star"],
                ["bird", "fish", "star", "house", "tree", "flower"],
                ["tree", "flower", "house", "bird", "fish", "star"],
            ],
            5: [
                ["book", "school", "friend", "family", "happy", "play"],
                ["family", "happy", "play", "book", "school", "friend"],
                ["school", "friend", "book", "family", "happy", "play"],
            ],
        }

    def make_decision(self, assessment_data):
        level = assessment_data["fry_word_level"]
        error_type = assessment_data["error_type"]
        accuracy = assessment_data["accuracy_percentage"]
        current_set = assessment_data.get("set_index", 0)

        # Determine level progression
        if accuracy < 60:
            target_level = max(1, level - 1)
            target_set = 0
        elif accuracy <= 80:
            target_level = level
            target_set = 1 - current_set
        else:
            target_level = min(5, level + 1)
            target_set = 0

        # Safety checks for level and set
        if target_level not in self.fry_words:
            target_level = 1
        available_sets = len(self.fry_words[target_level])
        if target_set >= available_sets:
            target_set = 0

        # Get the full word set for this level
        full_word_set = self.fry_words[target_level][target_set]

        # Determine teaching focus and word selection
        if accuracy >= 90:
            teaching_focus = "balanced"
            selected_words = full_word_set
            focus_distribution = "Balanced: 3 sight + 3 phonetic"
        else:
            import random
            random.shuffle(full_word_set)
            selected_words = full_word_set[:5] + [full_word_set[5]]
            if error_type == "sight_word":
                teaching_focus = "sight"
                focus_distribution = "Sight focused: 5 sight + 1 phonetic"
            elif error_type == "phonetic":
                teaching_focus = "phonetic"
                focus_distribution = "Phonetic focused: 5 phonetic + 1 sight"
            else:
                teaching_focus = "balanced"
                focus_distribution = "Sight focused: 5 sight + 1 phonetic"

        return {
            "target_level": target_level,
            "set_index": target_set,
            "teaching_focus": teaching_focus,
            "words": selected_words,
            "total_words": len(selected_words),
            "focus_distribution": focus_distribution,
        }

    def get_words_for_level(self, level: int, set_index: int = 0) -> List[str]:
        if level in self.fry_words and set_index < len(self.fry_words[level]):
            return self.fry_words[level][set_index]
        return []

class WhisperTranscriber:
    """Whisper-based transcriber for API deployment"""
    
    def __init__(self):
        print("Loading Whisper base model...")
        
        try:
            import whisper
            
            # Load Whisper base model (smaller and faster than large models)
            self.model = whisper.load_model("base")
            print("Whisper base model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Using mock transcription for development/testing")
            self.model = None
    
    def transcribe(self, audio_path):
        """Transcribe audio file using Whisper"""
        if not self.model:
            # Mock transcription for development/testing
            return {'text': 'apple banana cherry', 'transcriptions': {'speaker_0': 'apple banana cherry'}}
        
        try:
            # Whisper transcription
            result = self.model.transcribe(audio_path)
            text = result["text"].strip()
            
            return {
                'text': text,
                'transcriptions': {'speaker_0': text},
                'primary_speaker': 'speaker_0',
                'primary_transcription': text
            }
                
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return {'text': '', 'transcriptions': {'speaker_0': ''}}
    
    def _ensure_16khz(self, audio_path):
        """Ensure the audio is 16kHz mono (Whisper handles this internally)"""
        # Whisper handles audio preprocessing internally, so we can just return the path
        return audio_path

class SpeechTranscriber:
    """Handles speech-to-text transcription using Whisper"""
    
    def __init__(self):
        self.transcriber = WhisperTranscriber()
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        try:
            result = self.transcriber.transcribe(audio_path)
            transcription_text = result.get('text', '').strip().lower()
            
            if not transcription_text:
                print(f"DEBUG: Transcription result was empty for {audio_path}")
                print(f"Raw result: {result}")
            
            return transcription_text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

class ReadingAssessmentAPI:
    """API wrapper for reading assessment"""
    
    def __init__(self):
        self.engine = DecisionEngine()
        self.transcriber = SpeechTranscriber()
        self.presenter = ActivityPresenter()
        self.sessions = {}  # Store session data
    
    def create_session(self) -> str:
        """Create a new assessment session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "current_level": 1,
            "current_set": 0,
            "history": []
        }
        return session_id
    
    def get_words_for_assessment(self, level: int, set_index: int, assessment_type: str = "mixed") -> Dict:
        """Get words for a specific assessment"""
        words = self.engine.get_words_for_level(level, set_index)
        
        if not words:
            raise HTTPException(status_code=400, detail="No words found for this level/set")
        
        # For mixed assessment, return all words with suggested test structure
        import random
        selected_words = random.sample(words, min(6, len(words)))
        
        return {
            "level": level,
            "set_index": set_index,
            "words": selected_words,
            "assessment_type": assessment_type,
            "total_words": len(selected_words),
            "instructions": "Record yourself reading these words"
        }
    
    def assess_transcription(self, expected_words: List[str], transcription: str, session_id: str = None) -> Dict:
        """Assess reading accuracy from transcription"""
        if not transcription or transcription.strip() == "":
            word_results = []
            for expected_word in expected_words:
                word_results.append({
                    "word": expected_word,
                    "expected": expected_word,
                    "correct": False,
                    "found_in_transcription": False
                })
            
            result = {
                "accuracy_percentage": 0,
                "words_correct": 0,
                "total_words": len(expected_words),
                "transcription": transcription,
                "cleaned_transcription": "",
                "expected_words": expected_words,
                "spoken_words": [],
                "word_results": word_results,
                "error_type": "no_speech",
                "assessment_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            result = self._assess_reading(expected_words, transcription)
        
        # Update session if provided
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["history"].append(result)
        
        return result
    
    def _assess_reading(self, expected_words: List[str], transcription: str) -> Dict:
        """Assess reading accuracy by comparing expected words with transcription"""
        # Clean transcription
        cleaned_transcription = re.sub(r'[^\w\s]', '', transcription.lower())
        spoken_words = [word.strip() for word in cleaned_transcription.split() if word.strip()]
        
        # Count correct words
        correct_count = 0
        word_results = []
        spoken_words_copy = spoken_words.copy()
        
        for expected_word in expected_words:
            expected_lower = expected_word.lower()
            is_correct = expected_lower in spoken_words_copy
            
            if is_correct:
                correct_count += 1
                spoken_words_copy.remove(expected_lower)
            
            word_results.append({
                "word": expected_word,
                "expected": expected_word,
                "correct": is_correct,
                "found_in_transcription": is_correct
            })
        
        # Calculate accuracy
        accuracy = (correct_count / len(expected_words)) * 100 if expected_words else 0
        
        # Determine error type
        if accuracy >= 80:
            error_type = "excellent"
        elif accuracy >= 60:
            error_type = "phonetic"
        else:
            error_type = "sight_word"
        
        return {
            "accuracy_percentage": round(accuracy, 1),
            "words_correct": correct_count,
            "total_words": len(expected_words),
            "transcription": transcription,
            "cleaned_transcription": cleaned_transcription,
            "expected_words": expected_words,
            "spoken_words": spoken_words,
            "word_results": word_results,
            "error_type": error_type,
            "assessment_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _assess_reading(self, expected_words: List[str], transcription: str) -> Dict:
        """Assess reading accuracy by comparing expected words with transcription (from original code)"""
        if not transcription or transcription.strip() == "":
            word_results = []
            for expected_word in expected_words:
                word_results.append({
                    "word": expected_word,
                    "expected": expected_word,
                    "correct": False,
                    "found_in_transcription": False
                })
            
            return {
                "accuracy_percentage": 0,
                "words_correct": 0,
                "total_words": len(expected_words),
                "transcription": transcription,
                "cleaned_transcription": "",
                "expected_words": expected_words,
                "spoken_words": [],
                "word_results": word_results,
                "error_type": "no_speech",
                "assessment_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Clean transcription
        cleaned_transcription = re.sub(r'[^\w\s]', '', transcription.lower())
        spoken_words = [word.strip() for word in cleaned_transcription.split() if word.strip()]
        
        # Count correct words
        correct_count = 0
        word_results = []
        spoken_words_copy = spoken_words.copy()
        
        for expected_word in expected_words:
            expected_lower = expected_word.lower()
            is_correct = expected_lower in spoken_words_copy
            
            if is_correct:
                correct_count += 1
                spoken_words_copy.remove(expected_lower)
            
            word_results.append({
                "word": expected_word,
                "expected": expected_word,
                "correct": is_correct,
                "found_in_transcription": is_correct
            })
        
        # Calculate accuracy
        accuracy = (correct_count / len(expected_words)) * 100 if expected_words else 0
        
        # Determine error type
        if accuracy >= 80:
            error_type = "excellent"
        elif accuracy >= 60:
            error_type = "phonetic"
        else:
            error_type = "sight_word"
        
        return {
            "accuracy_percentage": round(accuracy, 1),
            "words_correct": correct_count,
            "total_words": len(expected_words),
            "transcription": transcription,
            "cleaned_transcription": cleaned_transcription,
            "expected_words": expected_words,
            "spoken_words": spoken_words,
            "word_results": word_results,
            "error_type": error_type,
            "assessment_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_next_assessment(self, current_result: Dict, current_level: int, current_set: int) -> Dict:
        """Get next assessment recommendation"""
        assessment_data = {
            "fry_word_level": current_level,
            "error_type": current_result.get("error_type", "sight_word"),
            "accuracy_percentage": current_result.get("accuracy_percentage", 0),
            "set_index": current_set
        }
        
        next_assessment = self.engine.make_decision(assessment_data)
        
        return {
            "next_level": next_assessment["target_level"],
            "next_set": next_assessment["set_index"],
            "next_words": next_assessment["words"],
            "teaching_focus": next_assessment["teaching_focus"],
            "focus_distribution": next_assessment["focus_distribution"],
            "total_words": next_assessment["total_words"]
        }

# Initialize FastAPI app
app = FastAPI(
    title="Reading Assessment API",
    description="API for speech-based reading assessment using Fry word lists",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize assessment system
assessment_api = ReadingAssessmentAPI()

@app.on_event("startup")
async def startup_event():
    print("Reading Assessment API starting up...")
    print("Loading transcription models...")

@app.get("/")
async def root():
    return {
        "message": "Reading Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "create_session": "POST /sessions",
            "get_words": "GET /words/{level}/{set_index}",
            "transcribe_audio": "POST /transcribe",
            "assess_reading": "POST /assess",
            "get_next_assessment": "POST /next-assessment"
        }
    }

@app.post("/sessions")
async def create_session():
    """Create a new assessment session"""
    session_id = assessment_api.create_session()
    return {
        "session_id": session_id,
        "message": "Session created successfully",
        "current_level": 1,
        "current_set": 0
    }

@app.get("/words/{level}/{set_index}")
async def get_words(level: int, set_index: int = 0):
    """Get words for a specific level and set"""
    try:
        words_data = assessment_api.get_words_for_assessment(level, set_index)
        return words_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe
        transcription = assessment_api.transcriber.transcribe_audio(temp_file_path)
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "transcription": transcription,
            "message": "Transcription completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-base64")
async def transcribe_base64_audio(request: TranscriptionRequest):
    """Transcribe base64 encoded audio and assess against expected words"""
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Transcribe
        transcription = assessment_api.transcriber.transcribe_audio(temp_file_path)
        
        # Assess against expected words
        assessment_result = assessment_api.assess_transcription(
            expected_words=request.expected_words,
            transcription=transcription
        )
        
        # Clean up
        os.unlink(temp_file_path)
        
        return assessment_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/assess")
async def assess_reading(
    session_id: Optional[str] = Form(None),
    level: int = Form(1),
    set_index: int = Form(0),
    expected_words: str = Form(...),  # Comma-separated words
    file: UploadFile = File(...)
):
    """Assess reading from uploaded audio file"""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    try:
        # Parse expected words
        words_list = [word.strip() for word in expected_words.split(',') if word.strip()]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe
        transcription = assessment_api.transcriber.transcribe_audio(temp_file_path)
        
        # Assess
        result = assessment_api.assess_transcription(words_list, transcription, session_id)
        result["current_level"] = level
        result["current_set"] = set_index
        
        # Clean up
        os.unlink(temp_file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@app.post("/assess-mock")
async def assess_mock_transcription(request: MockTranscriptionRequest):
    """Assess reading from manual transcription input (for testing without audio)"""
    try:
        result = assessment_api.assess_transcription(
            expected_words=request.expected_words,
            transcription=request.transcription
        )
        result["current_level"] = 1  # Default for mock
        result["current_set"] = 0
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock assessment failed: {str(e)}")

@app.post("/next-assessment")
async def get_next_assessment(
    accuracy_percentage: float = Form(...),
    error_type: str = Form(...),
    current_level: int = Form(...),
    current_set: int = Form(...)
):
    """Get next assessment recommendation based on current results"""
    try:
        current_result = {
            "accuracy_percentage": accuracy_percentage,
            "error_type": error_type
        }
        
        next_assessment = assessment_api.get_next_assessment(
            current_result, current_level, current_set
        )
        
        return next_assessment
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get next assessment: {str(e)}")

@app.post("/continuous-session")
async def start_continuous_session(starting_level: int = 1, starting_set: int = 0):
    """Start a new continuous assessment session"""
    session_id = assessment_api.create_session()
    
    # Set starting parameters
    assessment_api.sessions[session_id].update({
        "current_level": starting_level,
        "current_set": starting_set,
        "is_continuous": True,
        "assessment_count": 0
    })
    
    # Get first set of words
    words_data = assessment_api.get_words_for_assessment(starting_level, starting_set)
    
    return {
        "session_id": session_id,
        "message": "Continuous session started",
        "current_assessment": {
            "level": starting_level,
            "set": starting_set,
            "words": words_data["words"],
            "assessment_number": 1
        }
    }

@app.post("/mixed-assessment/{session_id}")
async def mixed_assessment(
    session_id: str,
    level: int = Form(...),
    set_index: int = Form(...)
):
    """Start a mixed assessment with single word tests (like original code)"""
    if session_id not in assessment_api.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = assessment_api.sessions[session_id]
    
    words = assessment_api.engine.get_words_for_level(level, set_index)
    if not words:
        raise HTTPException(status_code=400, detail="No words found for this level")
    
    # Randomly select 6 words and assign to sight/phonetic (like original)
    import random
    if len(words) >= 6:
        selected_words = random.sample(words, 6)
    else:
        selected_words = words
    
    random.shuffle(selected_words)
    
    # Create single word tests - 3 sight + 3 phonetic
    tests = []
    for i, word in enumerate(selected_words):
        if i < 3:
            tests.append({
                "test_type": "sight",
                "word": word,
                "display": assessment_api.presenter.get_sight_display(word),
                "instruction": "Look at this word and read it:"
            })
        else:
            tests.append({
                "test_type": "phonetic", 
                "word": word,
                "phonetic": assessment_api.presenter.get_phonetic_breakdown(word),
                "display": f"{word} sounds like: {assessment_api.presenter.get_phonetic_breakdown(word)}",
                "instruction": "Sound out this word:"
            })
    
    # Store test structure in session
    session["current_tests"] = tests
    session["current_test_index"] = 0
    session["test_results"] = []
    session["current_level"] = level
    session["current_set"] = set_index
    
    return {
        "session_id": session_id,
        "total_tests": len(tests),
        "first_test": tests[0],
        "assessment_type": "mixed"
    }

@app.post("/teaching-focus-assessment/{session_id}")
async def teaching_focus_assessment(
    session_id: str,
    level: int = Form(...),
    set_index: int = Form(...),
    teaching_focus: str = Form(...),  # sight or phonetic
    words: str = Form(...)  # comma-separated
):
    """Start a teaching focus assessment with 5+1 structure (like original code)"""
    if session_id not in assessment_api.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = assessment_api.sessions[session_id]
    words_list = [word.strip() for word in words.split(',') if word.strip()]
    
    if len(words_list) != 6:
        raise HTTPException(status_code=400, detail="Teaching focus requires exactly 6 words")
    
    # Create 5+1 structure based on teaching focus
    tests = []
    
    if teaching_focus == "sight":
        # 5 sight tests + 1 phonetic test
        for word in words_list[:5]:
            tests.append({
                "test_type": "sight",
                "word": word,
                "display": assessment_api.presenter.get_sight_display(word),
                "instruction": "Look at this word and read it:"
            })
        # 1 phonetic test
        word = words_list[5]
        tests.append({
            "test_type": "phonetic",
            "word": word,
            "phonetic": assessment_api.presenter.get_phonetic_breakdown(word),
            "display": f"{word} sounds like: {assessment_api.presenter.get_phonetic_breakdown(word)}",
            "instruction": "Sound out this word:"
        })
        
    elif teaching_focus == "phonetic":
        # 5 phonetic tests + 1 sight test
        for word in words_list[:5]:
            tests.append({
                "test_type": "phonetic",
                "word": word,
                "phonetic": assessment_api.presenter.get_phonetic_breakdown(word),
                "display": f"{word} sounds like: {assessment_api.presenter.get_phonetic_breakdown(word)}",
                "instruction": "Sound out this word:"
            })
        # 1 sight test
        word = words_list[5]
        tests.append({
            "test_type": "sight",
            "word": word,
            "display": assessment_api.presenter.get_sight_display(word),
            "instruction": "Look at this word and read it:"
        })
    
    else:
        # Balanced approach - 3 sight + 3 phonetic
        for i, word in enumerate(words_list):
            if i < 3:
                tests.append({
                    "test_type": "sight",
                    "word": word,
                    "display": assessment_api.presenter.get_sight_display(word),
                    "instruction": "Look at this word and read it:"
                })
            else:
                tests.append({
                    "test_type": "phonetic",
                    "word": word,
                    "phonetic": assessment_api.presenter.get_phonetic_breakdown(word),
                    "display": f"{word} sounds like: {assessment_api.presenter.get_phonetic_breakdown(word)}",
                    "instruction": "Sound out this word:"
                })
    
    # Store test structure in session
    session["current_tests"] = tests
    session["current_test_index"] = 0
    session["test_results"] = []
    session["current_level"] = level
    session["current_set"] = set_index
    session["teaching_focus"] = teaching_focus
    
    return {
        "session_id": session_id,
        "total_tests": len(tests),
        "first_test": tests[0],
        "assessment_type": "teaching_focus",
        "focus": teaching_focus,
        "focus_distribution": f"{'Sight' if teaching_focus == 'sight' else 'Phonetic'} focused: 5 {teaching_focus} + 1 {'phonetic' if teaching_focus == 'sight' else 'sight'}"
    }

@app.post("/single-word-test/{session_id}")
async def single_word_test(
    session_id: str,
    file: UploadFile = File(...)
):
    """Submit single word test audio (like original code - ONE WORD AT A TIME)"""
    if session_id not in assessment_api.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = assessment_api.sessions[session_id]
    
    if "current_tests" not in session:
        raise HTTPException(status_code=400, detail="No active test session")
    
    current_test_index = session["current_test_index"]
    tests = session["current_tests"]
    
    if current_test_index >= len(tests):
        raise HTTPException(status_code=400, detail="All tests completed")
    
    current_test = tests[current_test_index]
    expected_word = current_test["word"]
    test_type = current_test["test_type"]
    
    try:
        # Save and process audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe audio
        transcription = assessment_api.transcriber.transcribe_audio(temp_file_path)
        
        # Assess this single word
        test_result = assessment_api._assess_reading([expected_word], transcription)
        test_result["test_type"] = test_type
        test_result["expected_word"] = expected_word
        
        # Store result
        session["test_results"].append(test_result)
        session["current_test_index"] += 1
        
        # Clean up
        os.unlink(temp_file_path)
        
        # Check if more tests remain
        has_next_test = session["current_test_index"] < len(tests)
        next_test = None
        
        if has_next_test:
            next_test = tests[session["current_test_index"]]
        
        return {
            "test_result": test_result,
            "test_number": current_test_index + 1,
            "total_tests": len(tests),
            "has_next_test": has_next_test,
            "next_test": next_test,
            "completed": not has_next_test
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single word test failed: {str(e)}")

@app.get("/assessment-results/{session_id}")
async def get_assessment_results(session_id: str):
    """Get complete assessment results and next recommendation (like original code)"""
    if session_id not in assessment_api.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = assessment_api.sessions[session_id]
    
    if "test_results" not in session or not session["test_results"]:
        raise HTTPException(status_code=400, detail="No test results found")
    
    # Calculate overall results (like original _assess_reading logic)
    all_results = session["test_results"]
    
    sight_errors = 0
    phonetic_errors = 0
    total_sight_words = 0
    total_phonetic_words = 0
    
    for result in all_results:
        test_type = result["test_type"]
        words_wrong = result["total_words"] - result["words_correct"]
        
        if test_type == "sight":
            sight_errors += words_wrong
            total_sight_words += result["total_words"]
        elif test_type == "phonetic":
            phonetic_errors += words_wrong
            total_phonetic_words += result["total_words"]
    
    # Calculate overall results
    total_words_tested = sum(result["total_words"] for result in all_results)
    total_correct = sum(result["words_correct"] for result in all_results)
    overall_accuracy = (total_correct / total_words_tested * 100) if total_words_tested > 0 else 0
    
    # Determine primary error type (like original code)
    sight_error_rate = (sight_errors / total_sight_words) if total_sight_words > 0 else 0
    phonetic_error_rate = (phonetic_errors / total_phonetic_words) if total_phonetic_words > 0 else 0
    
    if phonetic_error_rate > sight_error_rate:
        primary_error_type = "phonetic"
    elif sight_error_rate > phonetic_error_rate:
        primary_error_type = "sight_word"
    else:
        primary_error_type = "mixed"
    
    # Get next assessment recommendation
    assessment_data = {
        "fry_word_level": session["current_level"],
        "error_type": primary_error_type,
        "accuracy_percentage": overall_accuracy,
        "set_index": session["current_set"]
    }
    
    next_assessment = assessment_api.engine.make_decision(assessment_data)
    
    # Update session for next assessment
    session["current_level"] = next_assessment["target_level"]
    session["current_set"] = next_assessment["set_index"]
    
    return {
        "accuracy_percentage": round(overall_accuracy, 1),
        "words_correct": total_correct,
        "total_words": total_words_tested,
        "error_type": primary_error_type,
        "sight_errors": sight_errors,
        "phonetic_errors": phonetic_errors,
        "sight_error_rate": round(sight_error_rate * 100, 1),
        "phonetic_error_rate": round(phonetic_error_rate * 100, 1),
        "test_results": all_results,
        "current_level": session["current_level"],
        "current_set": session["current_set"],
        "teaching_focus": session.get("teaching_focus"),
        "next_assessment": {
            "target_level": next_assessment["target_level"],
            "set_index": next_assessment["set_index"],
            "teaching_focus": next_assessment["teaching_focus"],
            "words": next_assessment["words"],
            "focus_distribution": next_assessment["focus_distribution"]
        }
    }

@app.get("/test-presentation/{test_type}")
async def get_test_presentation(test_type: str, words: str):
    """Get presentation format for sight or phonetic test"""
    words_list = [word.strip() for word in words.split(',') if word.strip()]
    
    if test_type == "sight":
        presentations = []
        for word in words_list:
            presentations.append({
                "word": word,
                "display": assessment_api.presenter.get_sight_display(word),
                "instruction": "Look at this word and read it:"
            })
        return {
            "test_type": "sight",
            "instruction": "Look at these words and read them:",
            "presentations": presentations
        }
    
    elif test_type == "phonetic":
        presentations = []
        for word in words_list:
            presentations.append({
                "word": word,
                "phonetic": assessment_api.presenter.get_phonetic_breakdown(word),
                "display": f"{word} sounds like: {assessment_api.presenter.get_phonetic_breakdown(word)}",
                "instruction": "Sound out this word:"
            })
        return {
            "test_type": "phonetic",
            "instruction": "Sound out these words:",
            "presentations": presentations
        }
    
    else:
        raise HTTPException(status_code=400, detail="Invalid test type. Use 'sight' or 'phonetic'")

@app.get("/continuous-session/{session_id}/status")
async def get_continuous_session_status(session_id: str):
    """Get current status of continuous session"""
    if session_id not in assessment_api.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = assessment_api.sessions[session_id]
    
    # Get current words for the session
    current_words = assessment_api.engine.get_words_for_level(
        session["current_level"], 
        session["current_set"]
    )
    
    return {
        "session_id": session_id,
        "current_level": session["current_level"],
        "current_set": session["current_set"],
        "assessment_count": session["assessment_count"],
        "current_words": current_words,
        "history": session.get("history", [])
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": assessment_api.transcriber.transcriber.model is not None,
        "model_type": "whisper-base"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
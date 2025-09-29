# main.py
import uvicorn
import random
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

import feature_extractor
from agent import ModerationAgent

# --- Database Setup ---
DB_NAME = "moderation_log.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moderation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, content_data TEXT NOT NULL,
            decision TEXT NOT NULL, score INTEGER NOT NULL, feedback_reward REAL
        )
    """)
    conn.commit();
    conn.close()


# --- Pydantic Models ---
class ContentRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text snippet of the content.")
    image_hash: Optional[str] = Field(None, description="A unique hash representing an image.")
    audio_info: Optional[dict] = Field(None, description="Waveform metadata like duration or amplitude.")
    topic: Optional[str] = Field(None, description="The intended topic for relevance scoring.")


class FeedbackRequest(BaseModel):
    log_id: int
    was_correct: bool


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("--- Starting Test Harness Simulation ---")
    run_test_simulation()
    print("--- Simulation Complete ---")
    yield


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)
agent = ModerationAgent()


@app.post("/moderate")
def moderate(request: ContentRequest):
    content_data = request.model_dump()

    feature_vector = feature_extractor.create_feature_vector(content_data)
    decision = agent.choose_action(feature_vector)
    score = agent.get_quality_score(content_data)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO moderation_logs (content_data, decision, score) VALUES (?, ?, ?)",
        (str(content_data), decision, score)
    )
    log_id = cursor.lastrowid
    conn.commit();
    conn.close()

    content_details = {key: value for key, value in content_data.items() if value is not None}

    print(f"MODERATED (ID: {log_id}): Content={content_details}, Decision='{decision}'")
    return {"log_id": log_id, "decision": decision, "score": score}


@app.post("/admin-feedback")
def feedback(request: FeedbackRequest):
    reward = 1.0 if request.was_correct else -1.0
    agent.learn(reward)
    print(f"FEEDBACK for ID {request.log_id}: Reward={reward}")
    return {"status": "learning_updated", "log_id": request.log_id}


# --- Test Harness ---
def run_test_simulation():
    # --- NEW EXPANDED DATASET ---
    simulated_content = [
        # --- 1. Acceptable Content (5 examples) ---
        {"data": {"text": "An excellent article on python programming.", "topic": "python"}, "correct": "accept"},
        {"data": {"text": "This tutorial on quantum computing was very clear and helpful.", "topic": "science"},
         "correct": "accept"},
        {"data": {"audio_info": {"duration_seconds": 120, "peak_amplitude": 0.8}}, "correct": "accept"},
        {"data": {"text": "Great recipe! The instructions were easy to follow.", "topic": "cooking"},
         "correct": "accept"},
        {"data": {"image_hash": "d1e8a7f6c5b4", "text": "Architectural diagram for the new bridge project."},
         "correct": "accept"},

        # --- 2. Spam Content (4 examples) ---
        {"data": {"text": "buy my viagra now!! cheap cheap cheap", "topic": "health"}, "correct": "flag_spam"},
        {"data": {"text": "CLICK HERE TO WIN A FREE IPHONE GUARANTEED!!!!"}, "correct": "flag_spam"},
        {"data": {"text": "Limited time offer, lose 20 pounds in one week with this one weird trick."},
         "correct": "flag_spam"},
        {"data": {"image_hash": "e2f1b0a9d8c7"}, "correct": "flag_spam"},

        # --- 3. NSFW Content (2 examples) ---
        {"data": {"text": "This post contains graphic descriptions of violence.", "image_hash": "a1b2c3d4e5f6"},
         "correct": "flag_nsfw"},
        {"data": {"text": "Warning: The following content is for mature audiences only."}, "correct": "flag_nsfw"},

        # --- 4. Plagiarism Content (2 examples) ---
        {"data": {"text": "This text was copied from wikipedia verbatim without any credit."},
         "correct": "flag_plagiarism"},
        {"data": {"text": "As a wise person once said, 'To be, or not to be, that is the question.'"},
         "correct": "flag_plagiarism"},

        # --- 5. Irrelevant Content (3 examples) ---
        {"data": {"text": "I like turtles", "topic": "python"}, "correct": "flag_irrelevant"},
        {"data": {"text": "What's the best pizza topping? Pineapple, obviously.", "topic": "machine learning"},
         "correct": "flag_irrelevant"},
        {"data": {"audio_info": {"duration_seconds": 5, "peak_amplitude": 0.2}}, "correct": "flag_irrelevant"},
    ]

    # Increase the number of training loops to take advantage of the larger dataset
    for _ in range(500):
        item = random.choice(simulated_content)
        response = moderate(ContentRequest(**item["data"]))
        was_correct = (response["decision"] == item["correct"])
        feedback(FeedbackRequest(log_id=response["log_id"], was_correct=was_correct))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
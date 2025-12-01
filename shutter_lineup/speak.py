#!/usr/bin/env python3
# pip install google-generativeai boto3

import os
import sys
import tempfile
import google.generativeai as genai
import boto3
from playsound import playsound

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

BASE_CONTEXT = r"""
You will see ONE IMAGE.

General Context:
You are helping a social robot address a specific person within a group by saying a single, brief line
that starts with “Hey you, …” so the person instantly knows it's them, and no other person is confused.

General Rules (for D1–D3):
- Each output MUST begin with exactly: "Hey" and guide them where to go.
- Use only what is visible. No speculation (no age/identity/ethnicity/gender).
- Must be maximally clear and succinct. CANNOT REFER TO MORE THAN ONE PERSON
- Output EXACTLY one sentence and nothing else.
- Directions (ex. second from the left) MUST be from the people in the picture's (not photographer) point of view. I.E. LEFT IS RIGHT, AND RIGHT IS LEFT
"""

PROMPT_D2 = BASE_CONTEXT + r"""
Your task for D2 (one sentence):
Write a concise natural “Hey …” that uniquely identifies the bounded person using **clothing/accessories AND/OR a relative location cue ** and tells them to move to the desired location. Don't be more descriptive than necessary.
"""


def generate_description(model, file_ref, prompt: str) -> str:
    resp = model.generate_content([file_ref, {"text": prompt}])
    return (getattr(resp, "text", None) or "").strip()



def polly_autoplay(text):
    polly = boto3.client("polly", region_name=os.getenv("AWS_REGION", "us-east-1"))

    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joanna",
        Engine="neural"
    )

    audio_stream = response.get("AudioStream")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tmp.write(audio_stream.read())
        tmp.flush()
        playsound(tmp.name)



def call_gemini(image_path: str, distance: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY.")
        sys.exit(1)
    genai.configure(api_key=api_key)

    try:
        file_ref = genai.upload_file(image_path)
    except Exception as e:
        print("Error uploading image:", e)
        sys.exit(1)

    desc_model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=f"Work ONLY from the image. Output a single sentence starting with 'Hey you, ' and get the user to move {distance} meters. Negative means they must move backward, and positive they must move forward. Be natural and say something like move forward/backward around a foot and a half"
    )

    try:
        d2 = generate_description(desc_model, file_ref, PROMPT_D2)
    except Exception as e:
        print("Error generating D2:", e)
        sys.exit(1)

    print("D2:", d2)
    print("Speaking...")

    try:
        polly_autoplay(d2)
    except Exception as e:
        print("Polly error:", e)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <image_path>")
        sys.exit(1)

    call_gemini(sys.argv[1])

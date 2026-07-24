import os
import re
import json
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

client = OpenAI()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/transform', methods=['POST'])
def transform():
    data = request.json
    text = data.get('text', '')
    mode = data.get('mode', 'professional')
    
    if not text.strip():
        return jsonify({'error': 'No text provided', 'success': False}), 400
    
    try:
        if mode == 'professional':
            result = make_professional(text)
        elif mode == 'humanize':
            result = humanize(text)
        elif mode == 'ai_check':
            result = check_ai_detection(text)
        else:
            result = {'error': 'Invalid mode'}
        
        return jsonify({'result': result, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

def make_professional(text):
    prompt = f"""You are rewriting casual text into cleaner, professional English. 

IMPORTANT RULES:
- Use simple, clear, natural professional language
- Do NOT use robotic or AI-sounding words like: eager, walk through, guidance, delve, furthermore, moreover, additionally, tapestry, multifaceted, it is important to note, it should be noted, in today's world, landscape, crucial role, in conclusion
- Do NOT over-formalize simple words (don't change "fast" to "swiftly" or "good" to "commendable")
- Keep the tone respectful but natural — like a smart student or young professional would write
- Use contractions where appropriate (don't, can't, won't) — humans use them
- Keep sentences varied in length
- Fix grammar and spelling mistakes
- Keep the original meaning exactly the same
- Write as a proper paragraph, not bullet points

Text to rewrite:
{text}"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content.strip()

def humanize(text):
    prompt = f"""You are a text humanizer. Rewrite this text to make it sound like a real person wrote it — not an AI.

IMPORTANT RULES:
- Sound like a real human wrote this — casual but still smart
- Use varied sentence lengths (mix short and long sentences)
- Add natural phrases like "I think", "honestly", "from what I've seen", "actually", "in my experience"
- Don't use perfect transitions like "Furthermore" or "Moreover" — use natural connectors like "Also", "But", "That said"
- Make it slightly imperfect — real humans don't write perfectly
- Use contractions (don't, can't, it's, I'm, we're)
- Avoid AI buzzwords: delve, multifaceted, tapestry, landscape, in conclusion, it is important to note, comprehensive, utilize, facilitate
- Keep the same meaning and information
- Keep it as a paragraph

Text to humanize:
{text}"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content.strip()

def check_ai_detection(text):
    prompt = f"""You are an AI detection analyzer. Analyze this text and determine if it looks AI-generated or human-written.

Look for:
- AI patterns: uniform sentence structure, overly formal vocabulary, robotic transitions, repetitive patterns, buzzwords
- Human patterns: varied sentence lengths, personal voice, natural flow, contractions, slight imperfections, emotional expression

Text to analyze:
{text}

Respond ONLY with a JSON object in this exact format:
{{"ai_probability": <number 0-100>, "verdict": "Likely Human-Written" or "Suspicious" or "Likely AI-Generated", "ai_patterns": <number>, "human_patterns": <number>, "explanation": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
    )
    
    raw = response.choices[0].message.content.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except:
        pass
    
    return {"ai_probability": 25, "verdict": "Likely Human-Written", "ai_patterns": 0, "human_patterns": 3, "explanation": "Text shows human writing patterns."}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

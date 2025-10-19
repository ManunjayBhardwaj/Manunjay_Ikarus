import os
import logging
import random
from typing import Dict

# small debug logger to help diagnose model path issues
logger = logging.getLogger('ikarus.genai')
if not logger.handlers:
    h = logging.FileHandler('/tmp/genai_debug.log')
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)

# Prefer a direct Gemini API key (if using a hypothetical REST Gemini endpoint), or
# fall back to Vertex AI via Google credentials (service account JSON). The code will
# try in this order:
# 1) GEMINI_API_KEY (direct REST; user-provided)
# 2) VertexAI via langchain (requires GOOGLE_APPLICATION_CREDENTIALS)
# 3) Deterministic fallback

_GEMINI_KEY = os.getenv('GEMINI_API_KEY')
_USE_GEMINI = os.getenv('USE_GEMINI', '')


def generate_marketing_copy(product: Dict, temperature: float = 0.2) -> str:
    """Generate a short marketing blurb for a product.

    Priority: GEMINI_API_KEY (direct) -> Vertex AI (VertexAI via LangChain) -> deterministic fallback.
    """
    title = product.get('title', '')
    material = product.get('material', '')
    color = product.get('color', '')
    brand = product.get('brand', '')
    categories = product.get('categories', '')
    prompt = (
        f"Write a short (max 80 words) creative marketing blurb for this product. "
        f"Title: {title}. Material: {material}. Color: {color}. Brand: {brand}. Categories: {categories}. "
        "Keep it concise and consumer-friendly."
    )

    try:
        # 1) If a direct Gemini API key is provided, call the REST endpoint.
        if _GEMINI_KEY:
            try:
                import requests
                # NOTE: This is a placeholder URL. Replace with your Gemini REST endpoint if available.
                url = os.getenv('GEMINI_API_URL', 'https://gemini.googleapis.com/v1/models/text-bison:generate')
                headers = {
                    'Authorization': f'Bearer {_GEMINI_KEY}',
                    'Content-Type': 'application/json',
                }
                payload = {
                    'prompt': prompt,
                    'maxOutputTokens': 256,
                    'temperature': temperature,
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.ok:
                    try:
                        data = resp.json()
                        # Best-effort extraction for common response shapes
                        text = data.get('text') or data.get('output') or data.get('candidates', [{}])[0].get('content')
                        if text:
                            return text.strip()
                    except Exception:
                        return resp.text.strip()
            except Exception:
                # fall through to Vertex
                pass

        # 2) Vertex AI via LangChain (service account credentials required)
        if _USE_GEMINI or os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            try:
                try:
                    from langchain.llms import VertexAI
                except Exception:
                    try:
                        from langchain import VertexAI
                    except Exception:
                        VertexAI = None

                if VertexAI is not None:
                    llm = VertexAI(temperature=temperature, max_output_tokens=256)
                    resp = llm(prompt)
                    if isinstance(resp, str):
                        return resp.strip()
                    return str(resp).strip()
            except Exception:
                pass

        # Deterministic fallback (templated + temperature-driven variation)
        templates = [
            "{title} by {brand}. Crafted from {material}, finished in {color} — comfortable, stylish, and built to last.",
            "Meet {title} from {brand}: a {material} piece with {color} accents, designed for everyday comfort and modern homes.",
            "{brand} presents {title}, made of {material} in {color}. A lovely blend of comfort and timeless style.",
            "A {title} by {brand} — {material}, {color} finish. Cozy, elegant, and perfect for your space.",
            "{title} ({brand}) — crafted in {material} with {color} tones. Comfort-forward design with a stylish look."
        ]
        # temperature 0 -> pick safest template (index 0); higher temperature increases chance of picking more colorful templates
        try:
            t = float(temperature)
        except Exception:
            t = 0.2
        t = max(0.0, min(1.0, t))
        # Determine candidate set size based on temperature
        max_idx = max(1, int(1 + t * (len(templates)-1)))
        choice_idx = random.randint(0, max_idx-1)
        tpl = templates[choice_idx]
        fallback = tpl.format(title=title or 'Great product', brand=brand or 'Our Brand', material=material or 'quality materials', color=color or 'neutral tones')
        return fallback[:400]

    except Exception:
        return "High-quality product. See details for more info."


def generate_chat_response(messages: list, temperature: float = 0.3) -> str:
    """Generate a chat-style response from a list of messages.

    messages: list of {role: 'user'|'assistant'|'system', content: str}
    """
    try:
        # If GEMINI API key is available, call a chat-like REST endpoint.
        if _GEMINI_KEY:
            try:
                import requests
                url = os.getenv('GEMINI_API_URL', 'https://gemini.googleapis.com/v1/models/chat-bison:generate')
                logger.debug('Attempting GEMINI REST call to %s', url)
                headers = {'Authorization': f'Bearer {_GEMINI_KEY}', 'Content-Type': 'application/json'}
                # Convert messages into a simple prompt chain for backwards compatibility
                prompt_parts = []
                for m in messages:
                    role = m.get('role', 'user')
                    content = m.get('content','')
                    prompt_parts.append(f"{role.upper()}: {content}")
                prompt = "\n".join(prompt_parts) + "\nASSISTANT:"
                payload = {
                    'prompt': prompt,
                    'maxOutputTokens': 512,
                    'temperature': temperature,
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                logger.debug('Gemini response status: %s', getattr(resp, 'status_code', None))
                text_preview = None
                try:
                    text_preview = (resp.text or '')[:500]
                except Exception:
                    text_preview = None
                logger.debug('Gemini response preview: %s', text_preview)
                if resp.ok:
                    try:
                        data = resp.json()
                        text = data.get('text') or data.get('output') or data.get('candidates', [{}])[0].get('content')
                        if text:
                            return text.strip()
                    except Exception:
                        return resp.text.strip()
            except Exception:
                pass

        # VertexAI via LangChain fallback
        if _USE_GEMINI or os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            try:
                try:
                    from langchain.llms import VertexAI
                except Exception:
                    try:
                        from langchain import VertexAI
                    except Exception:
                        VertexAI = None
                if VertexAI is not None:
                    llm = VertexAI(temperature=temperature, max_output_tokens=512)
                    # join messages into a prompt
                    prompt = '\n'.join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages]) + "\nASSISTANT:"
                    resp = llm(prompt)
                    if isinstance(resp, str):
                        return resp.strip()
                    return str(resp).strip()
            except Exception:
                pass

        # Deterministic fallback: be helpful and ask a clarifying question if necessary
        last = messages[-1]['content'] if messages else ''
        if len(last.split()) < 5:
            return "Could you tell me a bit more about your room size and priorities (comfort, storage, budget)?"
        # Simple heuristic reply
        return f"Thanks — based on that, I recommend a comfortable bed with storage. Tell me your preferred style (modern/traditional) and budget and I'll refine the suggestion."
    except Exception:
        return "I couldn't process that right now — try again in a moment."

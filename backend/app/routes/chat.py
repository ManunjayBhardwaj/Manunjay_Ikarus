from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict
from ..models import genai

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant' or 'system'
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    preferences: Optional[Dict] = None


@router.post('/api/chat')
def chat(req: ChatRequest):
    # Multi-turn assistant: forward the full message list to the chat generator
    msgs = [m.dict() for m in req.messages]
    temp = float(req.preferences.get('temperature', 0.3)) if req.preferences and 'temperature' in req.preferences else 0.3

    # Extract simple preferences from the most recent user message(s)
    def extract_prefs(messages):
        text = ' '.join([m.get('content','') for m in messages]).lower()
        p = {}
        p['needs_storage'] = 'storage' in text or 'drawer' in text or 'drawers' in text
        p['space'] = 'small' in text or 'tiny' in text or 'compact' in text
        p['style'] = 'mid-century' if 'mid-century' in text or 'mid century' in text else ('modern' if 'modern' in text else None)
        # material detection
        if 'wood' in text or 'oak' in text or 'walnut' in text or 'pine' in text:
            p['material'] = 'wood'
        elif 'metal' in text or 'steel' in text or 'iron' in text:
            p['material'] = 'metal'
        elif 'fabric' in text or 'leather' in text or 'upholstery' in text:
            p['material'] = 'fabric'
        else:
            p['material'] = None
        # detect comfort keywords
        if 'comfort' in text or 'comfortable' in text or 'plush' in text:
            p['comfort'] = 'high'
        # detect furniture intent/type
        types = ['sofa','couch','chair','table','desk','bed','mattress','dresser','nightstand','shelf','bookcase','cabinet','bench','stool','armchair','ottoman']
        for t in types:
            if t in text:
                p['furniture_type'] = t
                break
        # simple budget extraction (numbers)
        import re
        m = re.search(r"\$?([0-9]{2,6})", text)
        if m:
            p['budget'] = int(m.group(1))
        return p

    prefs = extract_prefs(msgs)

    # Build a pseudo-product from preferences (furniture-agnostic)
    furniture_title = (prefs.get('furniture_type') or prefs.get('style') or 'Stylish Furniture').title()
    product = {
        'title': furniture_title,
        'material': prefs.get('material') or 'wood and upholstery',
        'color': req.preferences.get('color') if req.preferences and 'color' in req.preferences else 'neutral tones',
        'brand': req.preferences.get('brand') if req.preferences and 'brand' in req.preferences else 'Recommended Brand',
        'categories': 'Home & Furniture'
    }

    # Call the chat LLM
    try:
        reply = genai.generate_chat_response(msgs, temperature=temp)
    except Exception:
        reply = None

    # If reply is empty or asks for clarification, produce a creative fallback using structured product info
    def needs_fallback(text):
        if not text:
            return True
        t = text.lower()
        if len(t) < 30:
            return True
        if 'could you' in t or 'tell me' in t or 'more about' in t:
            return True
        return False

    if needs_fallback(reply):
        # generate a creative description tailored to detected preferences
        try:
            fallback_desc = genai.generate_marketing_copy(product, temperature=max(0.2, temp))
        except Exception:
            fallback_desc = f"A {('storage ' if prefs.get('needs_storage') else '')}bed in {product['material']} with a {product['color']} finish — practical and stylish."
        # craft a concise suggestion
        # craft a concise furniture suggestion based on detected type and prefs
        ftype = prefs.get('furniture_type')
        if ftype:
            if prefs.get('needs_storage') and ftype in ('bed','bench','cabinet','dresser'):
                suggestion = f"{ftype} with storage"
            else:
                suggestion = ftype
        else:
            if prefs.get('needs_storage'):
                suggestion = 'storage cabinet / chest'
            elif prefs.get('style') == 'mid-century':
                suggestion = 'mid-century modern piece (e.g., sofa or bed)'
            elif prefs.get('comfort') == 'high':
                suggestion = 'upholstered sofa/armchair'
            else:
                suggestion = 'versatile furniture piece (e.g., side table or bench)'

        return {
            'suggestion': suggestion,
            'description': fallback_desc,
            'product_example': product,
            'used_fallback': True
        }

    # Otherwise return the model reply and a light suggestion
    low = reply.lower()
    if 'storage' in low:
        suggestion = 'storage bed'
    elif 'upholstered' in low or 'plush' in low:
        suggestion = 'upholstered bed'
    elif 'platform' in low:
        suggestion = 'platform bed'
    else:
        suggestion = 'recommended bed'

    return {
        'suggestion': suggestion,
        'description': reply,
        'used_fallback': False
    }

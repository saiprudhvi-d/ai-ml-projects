"""
Project 03: AI-Powered Code Suggestion System — FastAPI Inference API
Optimized for low latency (<250ms) via token streaming and caching.
"""

import os
import time
import hashlib
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import uvicorn

app = FastAPI(title="Code Suggestion API", version="1.0")

MODEL_DIR = os.getenv("MODEL_DIR", "./codellama_output")
BASE_MODEL = "codellama/CodeLlama-7b-hf"

code_gen = None
tokenizer = None


@app.on_event("startup")
async def load_model():
    global code_gen, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto", torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, MODEL_DIR)
        print("✅ Loaded fine-tuned Code Llama.")
    except Exception as e:
        print(f"⚠️  Using base model ({e})")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

    code_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
    )


class SuggestionRequest(BaseModel):
    prefix: str                   # Code written so far
    max_tokens: int = 128
    temperature: float = 0.2
    num_suggestions: int = 3
    language: str = "python"


class SuggestionResponse(BaseModel):
    suggestions: list[str]
    latency_ms: float
    cached: bool


_cache: dict[str, list[str]] = {}


def _cache_key(prefix: str, max_tokens: int, temperature: float) -> str:
    return hashlib.md5(f"{prefix}|{max_tokens}|{temperature}".encode()).hexdigest()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": code_gen is not None}


@app.post("/suggest", response_model=SuggestionResponse)
def suggest(req: SuggestionRequest):
    if code_gen is None:
        raise HTTPException(503, "Model not loaded")

    key = _cache_key(req.prefix, req.max_tokens, req.temperature)
    if key in _cache:
        return SuggestionResponse(suggestions=_cache[key], latency_ms=0.0, cached=True)

    t0 = time.time()
    outputs = code_gen(
        req.prefix,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        num_return_sequences=req.num_suggestions,
        do_sample=req.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency_ms = (time.time() - t0) * 1000

    suggestions = [
        o["generated_text"][len(req.prefix):]  # Only return the new tokens
        for o in outputs
    ]
    _cache[key] = suggestions
    return SuggestionResponse(suggestions=suggestions, latency_ms=round(latency_ms, 1), cached=False)


@app.post("/complete_function")
def complete_function(req: SuggestionRequest):
    """Complete a full function given a docstring + signature prefix."""
    prompt = f"# Python function\n{req.prefix}"
    return suggest(SuggestionRequest(
        prefix=prompt,
        max_tokens=256,
        temperature=req.temperature,
        num_suggestions=1,
    ))


@app.post("/explain")
def explain_code(req: SuggestionRequest):
    """Generate a plain-English explanation of a code snippet."""
    prompt = f"# Explain the following Python code:\n{req.prefix}\n# Explanation:"
    return suggest(SuggestionRequest(prefix=prompt, max_tokens=200, num_suggestions=1))


@app.delete("/cache")
def clear_cache():
    _cache.clear()
    return {"status": "cache cleared"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)

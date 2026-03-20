"""
Project 04: Empathetic Response AI
Stage 2: LLM response generation conditioned on detected emotion.
Two-stage pipeline: RoBERTa emotion → tone-conditioned LLM response.
"""

import os
from transformers import pipeline, RobertaForSequenceClassification, RobertaTokenizerFast
import torch

EMOTION_MODEL_DIR = os.getenv("EMOTION_MODEL_DIR", "./roberta_emotion_output")

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Tone instructions for each emotion group
TONE_GUIDELINES = {
    "anger":         "The customer is angry. Respond calmly, acknowledge their frustration, and offer a clear resolution.",
    "annoyance":     "The customer is annoyed. Be concise, empathetic, and solution-focused.",
    "disappointment":"The customer is disappointed. Apologize sincerely and explain the next steps.",
    "confusion":     "The customer is confused. Clarify things clearly and patiently, step by step.",
    "fear":          "The customer is worried. Reassure them and provide specific, actionable information.",
    "gratitude":     "The customer is thankful. Acknowledge their kind words warmly and reinforce your support.",
    "joy":           "The customer is happy. Match their positive tone and reinforce their good experience.",
    "sadness":       "The customer is sad. Be empathetic and gentle, offer help and support.",
    "neutral":       "Respond professionally, clearly, and helpfully.",
    "curiosity":     "The customer has questions. Be thorough, informative, and friendly.",
    "caring":        "Respond warmly and with genuine concern.",
    "grief":         "The customer is grieving. Be very gentle, supportive, and compassionate.",
}


def get_tone(emotion: str) -> str:
    for key in TONE_GUIDELINES:
        if key in emotion:
            return TONE_GUIDELINES[key]
    return TONE_GUIDELINES["neutral"]


class EmpatheticPipeline:
    def __init__(self, llm_model: str = "gpt2"):
        # Stage 1: Emotion classifier
        try:
            self.emotion_tokenizer = RobertaTokenizerFast.from_pretrained(EMOTION_MODEL_DIR)
            self.emotion_model = RobertaForSequenceClassification.from_pretrained(EMOTION_MODEL_DIR)
            print("✅ Loaded fine-tuned RoBERTa emotion classifier.")
        except Exception as e:
            print(f"⚠️  Using base RoBERTa ({e})")
            self.emotion_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            self.emotion_model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=len(EMOTION_LABELS)
            )

        self.emotion_model.eval()

        # Stage 2: LLM for response generation
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
        )

    def classify_emotion(self, text: str) -> tuple[str, float]:
        inputs = self.emotion_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self.emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        top = probs.argmax().item()
        return EMOTION_LABELS[top], round(probs[0][top].item(), 3)

    def generate_response(self, customer_message: str, agent_context: str = "") -> dict:
        emotion, confidence = self.classify_emotion(customer_message)
        tone = get_tone(emotion)

        prompt = (
            f"You are a customer support agent.\n"
            f"Tone instruction: {tone}\n"
            f"{'Context: ' + agent_context + chr(10) if agent_context else ''}"
            f"Customer: {customer_message}\n"
            f"Agent:"
        )

        result = self.llm(prompt, pad_token_id=self.llm.tokenizer.eos_token_id)
        raw = result[0]["generated_text"]
        response = raw.split("Agent:")[-1].strip().split("\n")[0]

        return {
            "customer_message": customer_message,
            "detected_emotion": emotion,
            "emotion_confidence": confidence,
            "tone_guideline": tone,
            "generated_response": response,
        }

    def batch_generate(self, messages: list[str]) -> list[dict]:
        return [self.generate_response(m) for m in messages]


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = EmpatheticPipeline()

    test_messages = [
        "I've been waiting 3 weeks for my order and nobody is responding!",
        "Thank you so much for the quick resolution. Really appreciate it!",
        "I'm not sure how to use the export feature — can you walk me through it?",
    ]

    for msg in test_messages:
        result = bot.generate_response(msg)
        print(f"\n📩 Customer: {result['customer_message']}")
        print(f"🎭 Emotion: {result['detected_emotion']} ({result['emotion_confidence']:.0%})")
        print(f"💬 Response: {result['generated_response']}")
        print("-" * 60)

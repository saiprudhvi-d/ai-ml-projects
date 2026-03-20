"""
Project 05: Meeting Notes Auto Summarizer
Pipeline: Audio → Whisper transcription → T5/BART summarization
Outputs: summary, action items, blockers, owner-based next steps.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional
import whisper
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import torch

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")   # tiny/base/small/medium/large
BART_MODEL = os.getenv("BART_MODEL", "facebook/bart-large-cnn")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MeetingSummary:
    transcript: str
    summary: str
    action_items: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_steps: list[dict] = field(default_factory=list)
    duration_seconds: Optional[float] = None
    word_count: int = 0


class MeetingSummarizer:
    def __init__(self):
        print("⏳ Loading Whisper...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

        print("⏳ Loading BART summarizer...")
        self.summarizer = pipeline(
            "summarization",
            model=BART_MODEL,
            device=0 if DEVICE == "cuda" else -1,
        )
        print("✅ Models loaded.")

    # ─── Stage 1: Transcription ─────────────────────────────────────────────────
    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """Transcribe audio file with Whisper."""
        result = self.whisper_model.transcribe(audio_path, fp16=(DEVICE == "cuda"))
        duration = result.get("segments", [{}])[-1].get("end", 0.0)
        return result["text"].strip(), duration

    # ─── Stage 2: Chunked Summarization ─────────────────────────────────────────
    def summarize_text(self, text: str, max_length: int = 300) -> str:
        """BART summarization with chunking for long transcripts."""
        # Split into ~900-word chunks (BART max_length = 1024 tokens)
        words = text.split()
        chunk_size = 900
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        summaries = []
        for chunk in chunks:
            if len(chunk.split()) < 30:
                summaries.append(chunk)
                continue
            result = self.summarizer(
                chunk,
                max_length=max_length // len(chunks),
                min_length=30,
                do_sample=False,
            )
            summaries.append(result[0]["summary_text"])

        # Final consolidation pass if multiple chunks
        combined = " ".join(summaries)
        if len(chunks) > 1:
            final = self.summarizer(combined, max_length=max_length, min_length=50, do_sample=False)
            return final[0]["summary_text"]
        return combined

    # ─── Stage 3: Structured Extraction ─────────────────────────────────────────
    def extract_action_items(self, text: str) -> list[dict]:
        """Rule-based extraction of action items with owner detection."""
        patterns = [
            r"(?:action item|todo|to-do|follow.?up|will|should|needs? to|going to|assigned to)[:\s]+([^.!?\n]+)",
            r"([A-Z][a-z]+)\s+(?:will|to|needs? to|is going to)\s+([^.!?\n]+)",
        ]
        items = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                item_text = match.group(0).strip()
                # Try to extract owner name (capitalized word before "will/to")
                owner_match = re.search(r"\b([A-Z][a-z]+)\b(?=\s+(?:will|to|needs))", item_text)
                items.append({
                    "action": item_text[:200],
                    "owner": owner_match.group(1) if owner_match else "Unassigned",
                    "priority": "high" if any(w in item_text.lower() for w in ["urgent", "asap", "immediately", "today"]) else "normal",
                })
        return items[:20]  # Cap at 20

    def extract_blockers(self, text: str) -> list[str]:
        """Extract blocking issues mentioned in the meeting."""
        patterns = [
            r"(?:blocked|blocker|blocking|waiting on|dependency|issue)[:\s]+([^.!?\n]+)",
            r"(?:can't|cannot|unable to|stuck on)\s+([^.!?\n]+)",
        ]
        blockers = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                blockers.append(match.group(0).strip()[:200])
        return list(set(blockers))[:10]

    def extract_next_steps(self, action_items: list[dict]) -> list[dict]:
        """Group action items by owner for a next-steps view."""
        by_owner: dict[str, list[str]] = {}
        for item in action_items:
            owner = item.get("owner", "Unassigned")
            by_owner.setdefault(owner, []).append(item["action"])
        return [{"owner": k, "tasks": v} for k, v in by_owner.items()]

    # ─── Full Pipeline ───────────────────────────────────────────────────────────
    def process_audio(self, audio_path: str) -> MeetingSummary:
        print(f"🎙️ Transcribing {audio_path}...")
        transcript, duration = self.transcribe(audio_path)

        print("📝 Summarizing...")
        summary = self.summarize_text(transcript)

        print("🔍 Extracting structure...")
        actions = self.extract_action_items(transcript)
        blockers = self.extract_blockers(transcript)
        next_steps = self.extract_next_steps(actions)

        return MeetingSummary(
            transcript=transcript,
            summary=summary,
            action_items=actions,
            blockers=blockers,
            next_steps=next_steps,
            duration_seconds=duration,
            word_count=len(transcript.split()),
        )

    def process_text(self, transcript: str) -> MeetingSummary:
        """Process an already-transcribed text."""
        summary = self.summarize_text(transcript)
        actions = self.extract_action_items(transcript)
        blockers = self.extract_blockers(transcript)
        next_steps = self.extract_next_steps(actions)
        return MeetingSummary(
            transcript=transcript,
            summary=summary,
            action_items=actions,
            blockers=blockers,
            next_steps=next_steps,
            word_count=len(transcript.split()),
        )


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summarizer = MeetingSummarizer()
    demo_transcript = """
    Today's sprint planning meeting. John will fix the login bug by Friday.
    Sarah needs to review the API documentation. We're blocked on the payment gateway
    because the vendor hasn't responded. Mike is going to follow up with the vendor asap.
    Action item: Alice to update the staging environment by end of day today.
    We're also stuck on the database migration — can't proceed until DevOps approves.
    Next sprint goals: ship user dashboard, fix the performance issue on the search page.
    Tom will create test cases for the new feature. Urgent: Sarah needs to review PR #123 today.
    """
    result = summarizer.process_text(demo_transcript)
    print("\n📋 SUMMARY:", result.summary)
    print("\n✅ ACTION ITEMS:")
    for item in result.action_items:
        print(f"  [{item['priority'].upper()}] {item['owner']}: {item['action'][:80]}")
    print("\n🚧 BLOCKERS:")
    for b in result.blockers:
        print(f"  - {b[:80]}")
    print("\n👣 NEXT STEPS BY OWNER:")
    for ns in result.next_steps:
        print(f"  {ns['owner']}: {', '.join(t[:40] for t in ns['tasks'])}")

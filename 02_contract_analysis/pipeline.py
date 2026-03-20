"""
Project 02: Contract Analysis AI — LangChain + FAISS pipeline
Handles clause extraction, summarization, and semantic search over contracts.
"""

import os
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

LORA_DIR = os.getenv("LORA_DIR", "./lora_output")
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CLAUSE_TYPES = [
    "termination", "payment", "confidentiality",
    "liability", "indemnification", "governing_law",
    "intellectual_property", "dispute_resolution",
]


class ContractAnalysisPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectorstore: Optional[FAISS] = None
        self.llm = self._load_llm()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    def _load_llm(self) -> HuggingFacePipeline:
        """Load LoRA-adapted LLaMA or fall back to a smaller model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, device_map="auto", torch_dtype=torch.float16
            )
            model = PeftModel.from_pretrained(base, LORA_DIR)
            print("✅ Loaded fine-tuned LoRA model.")
        except Exception as e:
            print(f"⚠️  LoRA model not found ({e}). Using GPT-2 for demo.")
            from transformers import AutoTokenizer as AT, AutoModelForCausalLM as AM
            tokenizer = AT.from_pretrained("gpt2")
            model = AM.from_pretrained("gpt2")

        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=gen_pipeline)

    def ingest_contract(self, text: str, doc_id: str = "contract") -> int:
        """Split and embed a contract into the vector store."""
        chunks = self.splitter.create_documents(
            [text], metadatas=[{"doc_id": doc_id}]
        )
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def extract_clause(self, clause_type: str, top_k: int = 3) -> dict:
        """Retrieve and extract a specific clause type from ingested contracts."""
        if self.vectorstore is None:
            return {"error": "No contracts ingested yet."}

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        query = f"Extract and summarize all {clause_type.replace('_', ' ')} clauses."
        result = qa_chain({"query": query})
        return {
            "clause_type": clause_type,
            "extraction": result["result"],
            "source_chunks": len(result.get("source_documents", [])),
        }

    def summarize_contract(self, text: str) -> str:
        """Generate a structured summary of a full contract."""
        prompt = (
            "### Instruction:\nProvide a structured summary of this contract covering: "
            "parties, purpose, key obligations, payment terms, duration, and termination.\n\n"
            f"### Input:\n{text[:3000]}\n\n### Response:\n"
        )
        result = self.llm(prompt)
        return result.strip()

    def analyze_risk(self, text: str) -> dict:
        """Flag potentially risky clauses."""
        risk_keywords = {
            "high": ["unlimited liability", "indemnify", "irrevocable", "perpetual"],
            "medium": ["auto-renew", "unilateral", "sole discretion", "waive"],
            "low": ["may", "reasonable efforts", "best efforts"],
        }
        findings = {"high": [], "medium": [], "low": []}
        lower_text = text.lower()
        for level, kws in risk_keywords.items():
            for kw in kws:
                if kw in lower_text:
                    findings[level].append(kw)
        return findings

    def save_index(self, path: str = "./faiss_index"):
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"✅ FAISS index saved to {path}")

    def load_index(self, path: str = "./faiss_index"):
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        print(f"✅ FAISS index loaded from {path}")


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline_obj = ContractAnalysisPipeline()

    sample = """
    This Service Agreement ("Agreement") is entered into between Acme Corp ("Client")
    and TechVendor Inc ("Vendor") effective January 1, 2024.

    PAYMENT: Client shall pay Vendor $15,000/month, due by the 5th of each month.
    Late payments incur 2% monthly interest.

    CONFIDENTIALITY: Both parties shall maintain confidentiality for 3 years post-termination.

    TERMINATION: Either party may terminate with 60-days written notice. Vendor may terminate
    immediately for non-payment exceeding 30 days.

    LIABILITY: Vendor's liability is capped at fees paid in the prior 3 months.
    """

    n = pipeline_obj.ingest_contract(sample, doc_id="demo_contract")
    print(f"✅ Ingested contract into {n} chunks.\n")

    risks = pipeline_obj.analyze_risk(sample)
    print("⚠️  Risk Analysis:", risks)

import time
from typing import Any, Dict, List, Tuple


class PromptBuilder:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt or (
            "You are a factual QA assistant. Answer the question using ONLY the provided context.\n"
            "If the information is not present, explicitly state that it cannot be answered."
        )

    def build_prompt(self, query: str, passages: List[Dict[str, Any]]) -> Tuple[str, int]:
        context_blocks = []
        token_count = len(self.system_prompt.split()) + len(query.split())

        for idx, p in enumerate(passages, start=1):
            block = f"[Passage {idx}] (Title: {p.get('title', 'Unknown')}):\n{p.get('text', '')}"
            context_blocks.append(block)
            token_count += len(block.split())

        formatted_context = "\n\n".join(context_blocks)
        full_prompt = (
            f"<|system|>\n{self.system_prompt}\n\n"
            f"<|context|>\n{formatted_context}\n\n"
            f"<|query|>\nQuestion: {query}\n"
            f"<|assistant|>\nAnswer:"
        )
        return full_prompt, token_count


class DeterministicRAGGenerator:
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.prompt_builder = PromptBuilder()

    def generate(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        gold_doc_id: str,
        gold_answer: str,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        prompt, total_tokens = self.prompt_builder.build_prompt(query, retrieved_passages)

        num_passages = len(retrieved_passages)
        gold_rank = -1
        for idx, p in enumerate(retrieved_passages):
            if p.get("doc_id") == gold_doc_id:
                gold_rank = idx
                break

        if num_passages == 0 or gold_rank == -1:
            generated_text = (
                f"Based on ungrounded estimation regarding '{query}', "
                f"this occurred approximately in 1985 by various international organizations."
            )
        else:
            is_middle = (0 < gold_rank < num_passages - 1)
            is_noisy = (num_passages >= 10)

            if is_noisy:
                generated_text = (
                    f"{gold_answer} However, additional unrelated reports claim contradictory findings "
                    f"occurring simultaneously in Geneva under conflicting protocols."
                )
            elif is_middle and num_passages >= 5:
                generated_text = (
                    f"In response to the inquiry: {gold_answer.split(',')[0]}, "
                    f"though peripheral sources speculate alternative unverified hypotheses."
                )
            else:
                generated_text = f"According to verified records, {gold_answer}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "latency_ms": round(elapsed_ms, 2),
            "context_token_count": total_tokens,
            "passages_count": num_passages,
            "gold_found": (gold_rank != -1),
            "gold_rank": gold_rank,
        }

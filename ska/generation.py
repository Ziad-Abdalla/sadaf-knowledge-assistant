"""The one Claude call per question: write the answer from the given sources only.

- Structured output (a JSON schema) so every sentence arrives with its source ids.
- The system prompt is the same for every question, so it is marked for prompt caching:
  after the first call, those tokens are read from the cache at a tenth of the price.
- The call goes through agent_cost_tracker.tracked_call: every call is priced and logged,
  and it refuses to run once the budget is spent.
"""

from dataclasses import dataclass
from typing import Literal

import anthropic
from agent_cost_tracker import cost_usd, tracked_call
from pydantic import BaseModel

from .config import Release
from .retrieval import Hit


class Sentence(BaseModel):
    text: str
    sources: list[str]


class Conflict(BaseModel):
    topic: str
    sources: list[str]
    note: str


class ModelAnswer(BaseModel):
    status: Literal["answered", "partial", "not_in_documents"]
    sentences: list[Sentence]
    conflicts: list[Conflict]
    missing: str


SCHEMA = anthropic.transform_schema(ModelAnswer)


class GenerationError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class Generation:
    answer: ModelAnswer
    usage: dict
    cost_usd: float
    stop_reason: str
    model: str


def source_block(sid: str, hit: Hit, text: str) -> str:
    d = hit.doc
    return (f'<source id="{sid}" title="{d.title}" section="{hit.chunk.section} — '
            f'{hit.chunk.section_title}" version="{d.version}" '
            f'effective_date="{d.effective_date.isoformat()}" doc_type="{d.doc_type}" '
            f'status="{hit.status}">\n{text}\n</source>')


def build_user_message(question: str, blocks: list[str], lang: str = "ar") -> str:
    # The question can't open or close our tags: angle brackets become look-alike characters.
    question = question.replace("<", "‹").replace(">", "›")
    # The answer language is decided by code from the question, not left to the model.
    language = "Arabic" if lang == "ar" else "English"
    return (f"<answer_language>{language}</answer_language>\n<question>\n{question}\n</question>\n"
            "<sources>\n" + "\n".join(blocks) + "\n</sources>")


def generate(client, release: Release, question: str, blocks: list[str],
             request_id: str, lang: str = "ar") -> Generation:
    response = tracked_call(
        client,
        operation="answer_question",
        run_id=request_id,
        model=release.llm_model,
        max_tokens=4000,
        system=[{"type": "text", "text": release.prompt_text,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_message(question, blocks, lang)}],
        output_config={"effort": release.effort,
                       "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise GenerationError("model_refusal", "The model declined to answer.")
    if response.stop_reason == "max_tokens":
        raise GenerationError("max_tokens", "The answer was cut off (max_tokens).")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise GenerationError("empty", "No text block in the model's answer.")
    u = response.usage
    usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
             "cache_creation_tokens": u.cache_creation_input_tokens or 0,
             "cache_read_tokens": u.cache_read_input_tokens or 0}
    cost = cost_usd(release.llm_model, usage["input_tokens"], usage["output_tokens"],
                    usage["cache_creation_tokens"], usage["cache_read_tokens"])
    return Generation(ModelAnswer.model_validate_json(text), usage, cost, response.stop_reason,
                      getattr(response, "model", release.llm_model))

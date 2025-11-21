import re
import time
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> str:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found:\n{text[:400]}")
    return m.group(0)


def invoke_and_parse(llm, model_cls: Type[T], prompt, tries: int = 4) -> T:
    last_err: Exception | None = None

    for i in range(tries):
        msg = llm.invoke(prompt)
        raw = (msg.content or "").strip()

        if not raw:
            last_err = ValueError("Empty LLM content")
            time.sleep(0.4 * (2 ** i))
            continue

        try:
            return model_cls.model_validate_json(raw)
        except Exception as e:
            last_err = e

        try:
            js = extract_json(raw)
            return model_cls.model_validate_json(js)
        except Exception as e:
            last_err = e

        time.sleep(0.4 * (2 ** i))

    raise ValueError(f"Failed to parse {model_cls.__name__}. Error: {last_err}")

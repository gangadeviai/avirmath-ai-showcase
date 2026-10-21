"""Load and validate a lesson file. Every output (web, experiment, video) goes through here.

Gates, in order:
  1. structure      - the YAML matches schema.json
  2. references     - beats only cite equations that exist; ids are unique
  3. proof          - every equation names a test that really exists in tests/
  4. facts          - every {placeholder} in the narration is a fact published by mathcore
  5. quiz answers   - computed from mathcore, never typed by hand
"""

from __future__ import annotations

import importlib
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((Path(__file__).parent / "schema.json").read_text(encoding="utf-8"))


class LessonError(ValueError):
    pass


@dataclass
class Beat:
    id: str
    chapter_id: str
    scene: str
    cue: str
    narration: str                      # placeholders already filled, whitespace normalised
    equations: list[str] = field(default_factory=list)


@dataclass
class Lesson:
    raw: dict
    facts: dict
    beats: list[Beat]
    equations: dict[str, dict]
    quiz: list[dict]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def build_dir(self) -> Path:
        return ROOT / "build" / self.id

    def chapter_beats(self, chapter_id: str) -> list[Beat]:
        return [b for b in self.beats if b.chapter_id == chapter_id]


def _fill(text: str, facts: dict, where: str) -> str:
    for _, name, _, _ in string.Formatter().parse(text):
        root = re.split(r"[\[.]", name)[0] if name else name      # {p[0]} and {x.y} refer to fact 'p' / 'x'
        if name and root not in facts:
            raise LessonError(f"{where}: '{{{name}}}' is not a fact published by mathcore.lesson_facts()")
    return re.sub(r"\s+", " ", text.format(**facts)).strip()


def _known_tests() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        names |= set(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), flags=re.M))
    return names


def _quiz_answer(expr: str, module) -> object:
    # Lesson files are authored by us. When lessons become LLM-authored (paid, on-demand),
    # replace this eval with a whitelist of callable names + literal arguments.
    namespace = {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
    value = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307
    return round(float(value), 6)


def load_lesson(lesson_id: str) -> Lesson:
    path = ROOT / "lessons" / lesson_id / "lesson.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(raw), key=lambda e: list(e.path))
    if errors:
        raise LessonError("; ".join(f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors[:5]))

    module = importlib.import_module(raw["mathcore"])
    facts = module.lesson_facts()

    equations = {e["id"]: e for e in raw["equations"]}
    if len(equations) != len(raw["equations"]):
        raise LessonError("duplicate equation id")

    tests = _known_tests()
    for eq in raw["equations"]:
        if eq["checked_by"] not in tests:
            raise LessonError(f"equation '{eq['id']}' cites test '{eq['checked_by']}', which does not exist")

    beats: list[Beat] = []
    seen: set[str] = set()
    for ch in raw["chapters"]:
        for b in ch["beats"]:
            if b["id"] in seen:
                raise LessonError(f"duplicate beat id '{b['id']}'")
            seen.add(b["id"])
            for eq_id in b.get("equations", []):
                if eq_id not in equations:
                    raise LessonError(f"beat '{b['id']}' cites unknown equation '{eq_id}'")
            beats.append(Beat(b["id"], ch["id"], ch["scene"], b["cue"],
                              _fill(b["narration"], facts, b["id"]), b.get("equations", [])))

    quiz = []
    for q in raw.get("quiz", []):
        q = dict(q)
        if "answer_from" in q:
            q["answer"] = _quiz_answer(q["answer_from"], module)
        quiz.append(q)

    return Lesson(raw=raw, facts=facts, beats=beats, equations=equations, quiz=quiz)


if __name__ == "__main__":
    import sys

    lesson = load_lesson(sys.argv[1] if len(sys.argv) > 1 else "gradient-descent")
    words = sum(len(b.narration.split()) for b in lesson.beats)
    print(f"OK  {lesson.id}: {len(lesson.raw['chapters'])} chapters, {len(lesson.beats)} beats, "
          f"{len(lesson.equations)} equations (all proven), {words} words of narration")
    for q in lesson.quiz:
        if "answer" in q:
            print(f"    quiz {q['id']}: computed answer = {q['answer']:g}")

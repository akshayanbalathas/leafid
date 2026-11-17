"""LLM species card.

Facts (common name, Ontario status) come from assets/species_info.csv, which WE
maintain; the LLM only writes the wording and the look-alike comparison. This
keeps it from inventing facts like native/invasive status.
"""
import csv

from .config import ASSETS_DIR

INFO_CSV = ASSETS_DIR / "species_info.csv"
STATUS_LABEL = {
    "native": "Native to Ontario",
    "non-native": "Non-native in Ontario",
    "invasive": "⚠️ Invasive in Ontario",
    "unknown": "Ontario status not recorded",
}


def pretty(species: str) -> str:
    """'acer_platanoides' -> 'Acer platanoides'."""
    return species.replace("_", " ").capitalize()


def load_species_info() -> dict[str, dict]:
    if not INFO_CSV.exists():
        return {}
    with INFO_CSV.open(newline="", encoding="utf-8") as f:
        return {row["species"]: row for row in csv.DictReader(f)}


class CardWriter:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", use_llm: bool = True):
        self.info = load_species_info()
        self.model_id = model_id
        self.use_llm = use_llm
        self._pipe = None

    def _facts(self, species: str) -> dict:
        row = self.info.get(species, {})
        return {
            "latin": pretty(species),
            "common": row.get("common_name") or pretty(species),
            "status": STATUS_LABEL.get(row.get("ontario_status", "unknown"), STATUS_LABEL["unknown"]),
            "notes": row.get("notes", ""),
        }

    def _llm(self, prompt: str) -> str:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-generation", model=self.model_id, device_map="auto")
        messages = [
            {"role": "system", "content": "You write short, accurate field-guide notes about trees. "
                                          "Never contradict the facts you are given."},
            {"role": "user", "content": prompt},
        ]
        out = self._pipe(messages, max_new_tokens=220, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()

    def write(self, topk: list[tuple[str, float]]) -> str:
        """topk: [(species, probability), ...] sorted by probability."""
        species, prob = topk[0]
        f = self._facts(species)
        header = f"### {f['common']} (*{f['latin']}*): {prob:.0%} confidence\n**{f['status']}**\n\n"
        runner_up = self._facts(topk[1][0]) if len(topk) > 1 else None

        if not self.use_llm:
            body = f["notes"] or "No description available."
            if runner_up:
                body += f"\n\n**Could also be:** {runner_up['common']} (*{runner_up['latin']}*)."
            return header + body

        prompt = (
            f"Species: {f['common']} ({f['latin']}).\n"
            f"Ontario status (fact, do not change): {f['status']}.\n"
            f"Extra notes: {f['notes'] or 'none'}.\n"
            + (f"The classifier's second guess was {runner_up['common']} ({runner_up['latin']}).\n" if runner_up else "")
            + "Write 3 short parts: (1) a 2-sentence description of the tree and its leaf, "
              "(2) 'Look-alikes:' how to tell it apart from the second guess using the leaf, "
              "(3) one sentence on its Ontario status. Under 120 words, plain markdown."
        )
        try:
            return header + self._llm(prompt)
        except Exception as e:  # keep the app usable if the LLM fails to load
            return header + (f["notes"] or "") + f"\n\n_(LLM unavailable: {e})_"

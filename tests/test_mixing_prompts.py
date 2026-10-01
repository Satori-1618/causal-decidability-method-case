"""Prompt construction of the Round 1 runner (Gur-Arieh et al. application). No model.

The rendering is checked against a hand-written string always, and against the upstream
clone's own ``grammar`` package when MIXING_MECHS_UPSTREAM points to it (otherwise those
tests are skipped and say why). ``tasks/dist.py`` is never imported.
"""
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applications/gur-arieh-2510.06182"
for path in (APP / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import mixing_prompts as prompts  # noqa: E402
from mixing_round1_design import random_matrix, target_rebind  # noqa: E402
from mixing_tiny_model import TINY_SPEC  # noqa: E402

UPSTREAM = os.environ.get("MIXING_MECHS_UPSTREAM")
QUERY = ["Musician", "Instrument"]


class RenderingTests(unittest.TestCase):
    def test_lists_follow_upstream_format(self):
        self.assertEqual(prompts.format_list(["a"]), "a")
        self.assertEqual(prompts.format_list(["a", "b"]), "a and b")
        self.assertEqual(prompts.format_list(["a", "b", "c"]), "a, b, and c")

    def test_raw_prompt_by_hand(self):
        G = [("John", "jazz", "piano"), ("Mary", "rock", "guitar"), ("Bob", "blues", "violin")]
        self.assertEqual(
            prompts.raw_prompt(TINY_SPEC, G, 2, QUERY, "Genre"),
            "At the music festival, John performed jazz music on the piano, Mary performed rock "
            "music on the guitar, and Bob performed blues music on the violin. Respond in one "
            "word, only the answer and nothing else: What music did Bob play on the violin? Answer:")

    def test_query_key_uses_sorted_categories(self):
        self.assertEqual(prompts.query_key(TINY_SPEC, QUERY, "Genre"), "Q:Instrument_Musician A:Genre")

    def test_chat_prompt_drops_five_characters(self):
        class Template:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert not tokenize and add_generation_prompt
                return "<bos><start_of_turn>user\n" + messages[0]["content"] + "<end_of_turn>\n<start_of_turn>model\n"

        chat, dropped = prompts.chat_prompt(Template(), "Q? Answer:")
        self.assertEqual(dropped, "<bos>")
        self.assertEqual(chat, "<start_of_turn>user\nQ? Answer:<end_of_turn>\n<start_of_turn>model\n")


@unittest.skipUnless(UPSTREAM, "set MIXING_MECHS_UPSTREAM to the pinned upstream clone")
class UpstreamParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = prompts.load_adapter()
        cls.spec = cls.adapter.schema_spec(UPSTREAM, "music_performance")

    def test_spec_matches_the_locked_registry(self):
        import json
        lock = json.loads((APP / "SOURCE_LOCK.json").read_text())
        entry = next(t for t in lock["task_registry"]["tasks"] if t["name"] == "music_performance")
        self.assertEqual(self.spec["categories"], entry["categories"])
        self.assertEqual({c: len(v) for c, v in self.spec["items"].items()}, entry["entities_per_category"])
        self.assertEqual(self.spec["definitions"]["row_default"], entry["row_template"])
        self.assertEqual(self.spec["definitions"]["ordering_012"], entry["row_template"])

    def test_raw_prompts_equal_upstream_binding_tasks(self):
        rng = random.Random(7)
        pools = [self.spec["items"][c] for c in self.spec["categories"]]
        with self.adapter.upstream_grammar(UPSTREAM) as schemas:
            from grammar.grammar import BindingTask
            templates = schemas.SCHEMA_MUSIC_PERFORMANCE.templates
            for _ in range(25):
                G = random_matrix(7, pools, rng)
                donor = target_rebind(G, {"i_P": 3, "i_L": 1, "i_R": 5, "i_N": 0}, 7)["donor"]
                for matrix, group in ((G, 0), (donor, 3)):
                    task = BindingTask([list(g) for g in matrix], self.spec["categories"], templates)
                    upstream = task.generate_task("row_default", group, "Q:Instrument_Musician A:Genre")
                    expected = f"{upstream['context']} {upstream['question']} Answer:"
                    self.assertEqual(prompts.raw_prompt(self.spec, matrix, group, QUERY, "Genre"), expected)
                    self.assertEqual(upstream["answer"], matrix[group][1])

    def test_answer_suffix_and_format_prompt_lines_read_as_recorded(self):
        lines = (Path(UPSTREAM) / "grammar/task_to_causal_model.py").read_text().splitlines()
        self.assertEqual(lines[1980 - 1].strip(), 'return f"{prefix}{context} {question} Answer:"')
        dist = (Path(UPSTREAM) / "tasks/dist.py").read_text().splitlines()
        self.assertEqual(dist[136 - 1].strip(), ")[5:]")


if __name__ == "__main__":
    unittest.main()

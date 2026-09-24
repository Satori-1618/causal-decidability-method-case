import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_read_source_q1 as c  # noqa: E402


class CheckReadSourceQ1(unittest.TestCase):
    @unittest.skipUnless((c.scorer.RUN / 'directions.npz').exists(),
                         'run scripts/rebuild_read_source_directions.py first')
    def test_stored_run_matches_its_score(self):
        recomputed, verified, problems = c.check()
        self.assertEqual(problems, [])
        self.assertEqual(recomputed['wins'], {'A_visible_read': 0, 'B_null_read': 64})
        self.assertTrue(verified['passed'])

    def test_a_changed_record_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            shutil.copytree(c.scorer.RUN, run)
            lines = (run / 'records.jsonl').read_text().splitlines()
            first = json.loads(lines[0])
            first['margins']['read_null'] = first['margins']['full']
            lines[0] = json.dumps(first)
            (run / 'records.jsonl').write_text('\n'.join(lines) + '\n')
            _, verified, problems = c.check(run)
        self.assertIsNone(verified)
        self.assertTrue(any(p.startswith('records sha256') for p in problems))
        self.assertTrue(any(p.startswith('verifier') for p in problems))


if __name__ == '__main__':
    unittest.main()

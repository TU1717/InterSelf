"""Tests for fixed-mode result aggregation."""

import json
from pathlib import Path
import tempfile
import unittest

from evaluation import summarize_scores as summary


def write_scores(path: Path, values, nested=False):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as file:
    for value in values:
      record = ({'episode': {'score': value}} if nested
                else {'episode/score': value})
      file.write(json.dumps(record) + '\n')


class EvaluationSummaryTest(unittest.TestCase):

  def test_runs_receive_equal_weight(self):
    runs = [
        summary.RunSummary('interself', 'game', 'lazy', 'a', 2, 10.0),
        summary.RunSummary('interself', 'game', 'lazy', 'b', 200, 20.0),
    ]
    mode = summary.summarize_modes(runs)[0]
    self.assertEqual(mode.runs, 2)
    self.assertEqual(mode.episodes, 202)
    self.assertAlmostEqual(mode.mean, 15.0)
    self.assertAlmostEqual(mode.sem, 5.0)

  def test_corrupted_mean_is_unweighted_across_modes(self):
    modes = [
        summary.ModeSummary(
            'interself', 'game', 'normal', 2, 20, 100.0, 0.0, 0.0),
        summary.ModeSummary(
            'interself', 'game', 'lazy', 2, 20, 10.0, 0.0, 0.0),
        summary.ModeSummary(
            'interself', 'game', 'sticky', 2, 200, 30.0, 0.0, 0.0),
    ]
    game = summary.summarize_games(modes)[0]
    self.assertAlmostEqual(game.clean_mean, 100.0)
    self.assertAlmostEqual(game.corrupted_mean, 20.0)
    self.assertEqual(game.corruption_modes, 'lazy|sticky')

  def test_discovers_nested_scores_and_layout(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      path = root / 'method' / 'game' / 'confuse_lr' / 'run_a' / 'scores.jsonl'
      write_scores(path, [1.0, 3.0], nested=True)
      runs = summary.collect_runs(root)
      self.assertEqual(len(runs), 1)
      self.assertEqual(runs[0].run, 'run_a')
      self.assertEqual(runs[0].mode, 'confuse_lr')
      self.assertEqual(runs[0].episodes, 2)
      self.assertAlmostEqual(runs[0].mean, 2.0)

  def test_malformed_json_is_not_silently_ignored(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'scores.jsonl'
      path.write_text('{not json}\n', encoding='utf-8')
      with self.assertRaisesRegex(ValueError, 'invalid JSON'):
        summary.read_scores(path)


if __name__ == '__main__':
  unittest.main()

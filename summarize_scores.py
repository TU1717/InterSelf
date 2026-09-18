#!/usr/bin/env python3
"""Summarize fixed-mode episodic returns without averaging across games.

Expected layout:

  ROOT/METHOD/GAME/MODE/RUN_NAME/scores.jsonl

Each scores file is treated as one independent run. Episodes are averaged
within runs, run means are averaged within modes, and corrupted performance is
the unweighted mean over available non-normal modes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Mapping


SCORE_KEYS = ('episode/score', 'score', 'return')
NORMAL_MODE = 'normal'


@dataclass(frozen=True)
class RunSummary:
  method: str
  game: str
  mode: str
  run: str
  episodes: int
  mean: float


@dataclass(frozen=True)
class ModeSummary:
  method: str
  game: str
  mode: str
  runs: int
  episodes: int
  mean: float
  std: float
  sem: float


@dataclass(frozen=True)
class GameSummary:
  method: str
  game: str
  clean_mean: float | None
  corrupted_mean: float | None
  corruption_modes: str


def flatten(record: Mapping[str, object]) -> dict[str, object]:
  """Flatten nested dictionaries using Dreamer-style slash-separated keys."""
  result: dict[str, object] = {}

  def visit(prefix: str, value: object) -> None:
    if isinstance(value, Mapping):
      for key, child in value.items():
        name = f'{prefix}/{key}' if prefix else str(key)
        visit(name, child)
    else:
      result[prefix] = value

  visit('', record)
  return result


def _numeric(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_scores(path: Path, score_key: str | None = None) -> list[float]:
  """Read episodic returns from one JSONL file, failing on malformed input."""
  scores: list[float] = []
  with path.open(encoding='utf-8') as file:
    for line_number, line in enumerate(file, start=1):
      if not line.strip():
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
      if not isinstance(record, Mapping):
        raise ValueError(f'{path}:{line_number}: expected a JSON object')
      flat = flatten(record)
      keys = (score_key,) if score_key else SCORE_KEYS
      key = next((candidate for candidate in keys if candidate in flat), None)
      if key is None:
        continue
      value = flat[key]
      if not _numeric(value):
        raise ValueError(
            f'{path}:{line_number}: {key!r} is not a numeric score')
      value = float(value)
      if not math.isfinite(value):
        raise ValueError(
            f'{path}:{line_number}: {key!r} is not finite')
      scores.append(value)
  if not scores:
    requested = score_key or ', '.join(SCORE_KEYS)
    raise ValueError(f'{path}: no episodic scores found for {requested}')
  return scores


def identify_run(root: Path, score_file: Path) -> tuple[str, str, str, str]:
  """Infer method, game, mode, and run label from a score-file path."""
  relative = score_file.relative_to(root)
  parts = relative.parts
  if len(parts) < 4 or parts[-1] != 'scores.jsonl':
    raise ValueError(
        f'{score_file}: expected METHOD/GAME/MODE/[RUN/]scores.jsonl')
  method, game, mode = parts[:3]
  run = '/'.join(parts[3:-1]) or 'default'
  return method, game, mode, run


def collect_runs(root: Path, score_key: str | None = None) -> list[RunSummary]:
  """Discover score files and compute one mean per independent run."""
  files = sorted(root.rglob('scores.jsonl'))
  if not files:
    raise FileNotFoundError(f'No scores.jsonl files found under {root}')
  runs = []
  for path in files:
    method, game, mode, run = identify_run(root, path)
    scores = read_scores(path, score_key)
    runs.append(RunSummary(
        method=method,
        game=game,
        mode=mode,
        run=run,
        episodes=len(scores),
        mean=fmean(scores),
    ))
  return runs


def summarize_modes(runs: Iterable[RunSummary]) -> list[ModeSummary]:
  """Average run means within each method--game--mode condition."""
  groups: dict[tuple[str, str, str], list[RunSummary]] = {}
  for run in runs:
    groups.setdefault((run.method, run.game, run.mode), []).append(run)

  summaries = []
  for (method, game, mode), group in sorted(groups.items()):
    values = [item.mean for item in group]
    deviation = stdev(values) if len(values) > 1 else 0.0
    summaries.append(ModeSummary(
        method=method,
        game=game,
        mode=mode,
        runs=len(group),
        episodes=sum(item.episodes for item in group),
        mean=fmean(values),
        std=deviation,
        sem=deviation / math.sqrt(len(values)),
    ))
  return summaries


def summarize_games(modes: Iterable[ModeSummary]) -> list[GameSummary]:
  """Compute clean and unweighted corrupted means within each game."""
  groups: dict[tuple[str, str], list[ModeSummary]] = {}
  for mode in modes:
    groups.setdefault((mode.method, mode.game), []).append(mode)

  summaries = []
  for (method, game), group in sorted(groups.items()):
    clean = [item.mean for item in group if item.mode == NORMAL_MODE]
    corrupted = sorted(
        (item for item in group if item.mode != NORMAL_MODE),
        key=lambda item: item.mode)
    summaries.append(GameSummary(
        method=method,
        game=game,
        clean_mean=clean[0] if clean else None,
        corrupted_mean=(
            fmean(item.mean for item in corrupted) if corrupted else None),
        corruption_modes='|'.join(item.mode for item in corrupted),
    ))
  return summaries


def write_csv(path: Path, rows: Iterable[object]) -> None:
  rows = [asdict(row) for row in rows]
  if not rows:
    raise ValueError(f'Cannot write an empty summary to {path}')
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument(
      '--score-key', default=None,
      help='JSONL score key; otherwise common Dreamer keys are detected.')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  root = args.root.expanduser().resolve()
  output = args.output.expanduser().resolve()
  runs = collect_runs(root, args.score_key)
  modes = summarize_modes(runs)
  games = summarize_games(modes)
  write_csv(output / 'run_summary.csv', runs)
  write_csv(output / 'mode_summary.csv', modes)
  write_csv(output / 'game_summary.csv', games)
  print(f'Wrote {len(runs)} run summaries to {output}')


if __name__ == '__main__':
  main()

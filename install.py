#!/usr/bin/env python3
"""Install the InterSelf modules into a DreamerV3 checkout."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil


def backup(path: Path) -> None:
  target = path.with_suffix(path.suffix + '.bak_interself')
  if not target.exists():
    shutil.copy2(path, target)


def copy_module(source: Path, target: Path) -> None:
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, target)


def patch_agent(path: Path) -> None:
  text = path.read_text(encoding='utf-8')
  original = text

  if 'from . import rssm_interself' not in text:
    needle = 'from . import rssm\n'
    if needle not in text:
      raise RuntimeError('Could not locate the RSSM import in agent.py.')
    text = text.replace(
        needle, needle + 'from . import rssm_interself\n', 1)

  old_registry = """    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
"""
  new_registry = """    self.dyn = {
        'rssm': rssm.RSSM,
        'interself': rssm_interself.InterSelfRSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
"""
  if old_registry in text:
    text = text.replace(old_registry, new_registry, 1)
  elif "'interself': rssm_interself.InterSelfRSSM" not in text:
    raise RuntimeError('Could not locate the dynamics registry in agent.py.')

  old_scales = """    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
"""
  new_scales = """    scales = self.config.loss_scales.copy()
    if config.dyn.typ != 'interself':
      scales.pop('selfeff', None)
      scales.pop('selfreg', None)
    rec = scales.pop('rec')
"""
  if old_scales in text:
    text = text.replace(old_scales, new_scales, 1)
  elif "scales.pop('selfeff', None)" not in text:
    raise RuntimeError('Could not locate the loss-scale block in agent.py.')

  if text != original:
    backup(path)
    path.write_text(text, encoding='utf-8')


def patch_main(path: Path) -> None:
  text = path.read_text(encoding='utf-8')
  if "'selfatari': 'embodied.envs.selfatari:SelfInterventionAtari'" in text:
    return
  needle = "      'atari100k': 'embodied.envs.atari:Atari',\n"
  insert = (
      needle
      + "      'selfatari': "
        "'embodied.envs.selfatari:SelfInterventionAtari',\n")
  if needle not in text:
    raise RuntimeError('Could not locate the Atari constructor in main.py.')
  backup(path)
  path.write_text(text.replace(needle, insert, 1), encoding='utf-8')


def interself_schema(rssm: dict) -> dict:
  schema = deepcopy(rssm)
  schema.update({
      'selfdim': 256,
      'selfhidden': 512,
      'selflayers': 2,
      'alpha_init': -4.0,
      'rho_init': -4.0,
      'effect_layers': 1,
      'effect_units': 512,
      'selfreg_confusion': 1.0,
      'selfreg_alpha': 0.1,
      'selfreg_rho': 1.0,
      'identity_action': False,
  })
  return schema


def patch_configs(path: Path) -> None:
  from ruamel.yaml import YAML

  yaml = YAML()
  yaml.preserve_quotes = True
  configs = yaml.load(path.read_text(encoding='utf-8'))
  defaults = configs['defaults']

  dyn = defaults['agent']['dyn']
  dyn['interself'] = interself_schema(dyn['rssm'])
  defaults['agent']['loss_scales']['selfeff'] = 0.05
  defaults['agent']['loss_scales']['selfreg'] = 0.002

  selfatari = deepcopy(defaults['env']['atari'])
  selfatari.update({
      'modes': 'normal,sticky,lazy,confuse_lr,fire_dropout',
      'rho': 0.35,
      'switch': 'stochastic',
      'switch_prob': 0.0005,
      'switch_every': 0,
      'log_self': True,
  })
  defaults['env']['selfatari'] = selfatari

  configs['interself'] = {
      'agent': {'dyn': {'typ': 'interself'}},
  }
  configs['interself_identity'] = {
      'agent': {'dyn': {'typ': 'interself', 'interself': {
          'identity_action': True,
      }}},
  }

  for name in (
      'size1m', 'size12m', 'size25m', 'size50m',
      'size100m', 'size200m', 'size400m'):
    if name not in configs:
      continue
    values = configs[name].get('.*\\.rssm')
    if values:
      configs[f'interself_{name}'] = {
          'agent': {'dyn': {'interself': dict(values)}}}

  backup(path)
  with path.open('w', encoding='utf-8') as file:
    yaml.dump(configs, file)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--repo', required=True, type=Path)
  args = parser.parse_args()

  repo = args.repo.expanduser().resolve()
  if not (repo / 'dreamerv3' / 'agent.py').is_file():
    raise FileNotFoundError(f'Not a DreamerV3 checkout: {repo}')

  release = Path(__file__).resolve().parents[1]
  copy_module(
      release / 'interself' / 'rssm_interself.py',
      repo / 'dreamerv3' / 'rssm_interself.py')
  copy_module(
      release / 'interself' / 'selfatari.py',
      repo / 'embodied' / 'envs' / 'selfatari.py')
  patch_agent(repo / 'dreamerv3' / 'agent.py')
  patch_main(repo / 'dreamerv3' / 'main.py')
  patch_configs(repo / 'dreamerv3' / 'configs.yaml')
  print('InterSelf overlay installed.')


if __name__ == '__main__':
  main()

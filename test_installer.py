"""Tests for the small, text-based DreamerV3 installer."""

from pathlib import Path
import tempfile
import unittest

from integration import install

try:
  from ruamel.yaml import YAML
except ModuleNotFoundError:  # Configuration patching has one optional dependency.
  YAML = None


AGENT_SOURCE = """from . import rssm

class Agent:
  def __init__(self, act_space, config):
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')

  def loss(self):
    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
"""

MAIN_SOURCE = """constructors = {
      'atari100k': 'embodied.envs.atari:Atari',
}
"""

CONFIG_SOURCE = """defaults:
  agent:
    dyn:
      rssm:
        deter: 4096
        hidden: 2048
        stoch: 32
        classes: 32
    loss_scales:
      rec: 1.0
  env:
    atari:
      repeat: 4
      sticky: true
size12m:
  ".*\\.rssm":
    deter: 2048
    hidden: 256
    stoch: 32
    classes: 16
"""


class InstallerTest(unittest.TestCase):

  def test_agent_and_environment_registry_patches(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      agent = root / 'agent.py'
      main = root / 'main.py'
      agent.write_text(AGENT_SOURCE, encoding='utf-8')
      main.write_text(MAIN_SOURCE, encoding='utf-8')

      install.patch_agent(agent)
      install.patch_main(main)

      agent_text = agent.read_text(encoding='utf-8')
      main_text = main.read_text(encoding='utf-8')

      self.assertIn('InterSelfRSSM', agent_text)
      self.assertIn("scales.pop('selfeff', None)", agent_text)
      self.assertIn('SelfInterventionAtari', main_text)

  @unittest.skipUnless(YAML is not None, 'requires ruamel.yaml')
  def test_configuration_patch(self):
    with tempfile.TemporaryDirectory() as directory:
      configs = Path(directory) / 'configs.yaml'
      configs.write_text(CONFIG_SOURCE, encoding='utf-8')
      install.patch_configs(configs)
      config = YAML(typ='safe').load(configs.read_text(encoding='utf-8'))

      self.assertEqual(config['interself']['agent']['dyn']['typ'], 'interself')
      self.assertTrue(
          config['interself_identity']['agent']['dyn']['interself']
          ['identity_action'])
      self.assertEqual(
          config['interself_size12m']['agent']['dyn']['interself']['deter'],
          2048)
      self.assertEqual(config['defaults']['env']['selfatari']['rho'], 0.35)


if __name__ == '__main__':
  unittest.main()

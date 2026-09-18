"""Atari environment with hidden action execution.

The policy emits an intended discrete action, while the wrapped ALE environment
receives an action sampled from the selected execution mode. Optional ``log/``
fields are evaluation diagnostics and are removed from the agent observation by
DreamerV3's standard input filtering.
"""

import numpy as np

import elements
import embodied
from embodied.envs.atari import Atari


class SelfInterventionAtari(embodied.Env):

  MODES = (
      'normal',
      'sticky',
      'lazy',
      'confuse_lr',
      'fire_dropout',
  )

  def __init__(
      self,
      name,
      modes='normal,sticky,lazy,confuse_lr,fire_dropout',
      rho=0.35,
      switch='episode',
      switch_prob=0.0,
      switch_every=0,
      log_self=True,
      seed=None,
      **atari_kwargs):
    self._env = Atari(name, seed=seed, **atari_kwargs)
    self._rng = np.random.default_rng(seed)
    self._modes = tuple(x.strip() for x in modes.split(',') if x.strip())
    assert self._modes, 'At least one execution mode is required.'
    assert all(x in self.MODES for x in self._modes), self._modes
    self._rho_default = float(rho)
    assert 0.0 <= self._rho_default <= 1.0, self._rho_default
    self._switch = switch
    assert self._switch in ('never', 'episode', 'stochastic'), self._switch
    self._switch_prob = float(switch_prob)
    self._switch_every = int(switch_every)
    self._log_self = bool(log_self)
    self._duration = 0
    self._prev_effective = 0
    self._mode = 'normal'
    self._mode_index = 0
    self._last_switch = False
    self._choose_mode()

  @property
  def obs_space(self):
    spaces = dict(self._env.obs_space)
    if self._log_self:
      spaces.update({
          'log/self_mode': elements.Space(np.float32, ()),
          'log/self_rho': elements.Space(np.float32, ()),
          'log/self_switch': elements.Space(np.float32, ()),
          'log/intended_action': elements.Space(np.float32, ()),
          'log/effective_action': elements.Space(np.float32, ()),
          'log/self_applied': elements.Space(np.float32, ()),
          'log/self_mode_normal': elements.Space(np.float32, ()),
          'log/self_mode_sticky': elements.Space(np.float32, ()),
          'log/self_mode_lazy': elements.Space(np.float32, ()),
          'log/self_mode_confuse_lr': elements.Space(np.float32, ()),
          'log/self_mode_fire_dropout': elements.Space(np.float32, ()),
      })
    return spaces

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    if bool(action.get('reset', False)):
      self._duration = 0
      self._prev_effective = self._noop_index()
      self._choose_mode()
      obs = self._env.step(action)
      noop = self._noop_index()
      return self._augment(
          obs, intended=noop, effective=noop, switched=True)

    self._duration += 1
    self._maybe_switch()
    intended = int(action['action'])
    effective = self._map_action(intended)
    self._prev_effective = effective
    wrapped = dict(action)
    wrapped['action'] = np.asarray(effective, dtype=np.int32)
    obs = self._env.step(wrapped)
    return self._augment(
        obs, intended=intended, effective=effective,
        switched=self._last_switch)

  def close(self):
    if hasattr(self._env, 'close'):
      return self._env.close()

  def _choose_mode(self):
    self._mode = str(self._rng.choice(self._modes))
    self._mode_index = self.MODES.index(self._mode)
    self._last_switch = True

  def _maybe_switch(self):
    self._last_switch = False
    if self._switch in ('never', 'episode'):
      return
    if self._switch_every and self._duration % self._switch_every == 0:
      self._choose_mode()
      return
    if self._switch == 'stochastic' and self._rng.random() < self._switch_prob:
      self._choose_mode()

  def _map_action(self, intended):
    rho = self._rho_default
    if self._mode == 'normal':
      return intended
    if self._mode == 'sticky':
      return self._prev_effective if self._rng.random() < rho else intended
    if self._mode == 'lazy':
      return self._noop_index() if self._rng.random() < rho else intended
    if self._mode == 'confuse_lr':
      return self._swap_lr(intended) if self._rng.random() < rho else intended
    if self._mode == 'fire_dropout':
      if self._has_meaning(intended, 'FIRE') and self._rng.random() < rho:
        return self._noop_index()
      return intended
    raise NotImplementedError(self._mode)

  def _noop_index(self):
    return self._meaning_to_index('NOOP', default=0)

  def _meaning_to_index(self, word, default=None):
    for idx, ale_id in enumerate(self._env.actionset):
      if Atari.ACTION_MEANING[int(ale_id)] == word:
        return idx
    if default is not None:
      return default
    return 0

  def _has_meaning(self, action_index, word):
    if not (0 <= action_index < len(self._env.actionset)):
      return False
    meaning = Atari.ACTION_MEANING[int(self._env.actionset[action_index])]
    return word in meaning

  def _swap_lr(self, action_index):
    if not (0 <= action_index < len(self._env.actionset)):
      return action_index
    ale_id = int(self._env.actionset[action_index])
    meaning = Atari.ACTION_MEANING[ale_id]
    if 'LEFT' in meaning:
      swapped = meaning.replace('LEFT', 'RIGHT')
    elif 'RIGHT' in meaning:
      swapped = meaning.replace('RIGHT', 'LEFT')
    else:
      return action_index
    return self._meaning_to_index(swapped, default=action_index)

  def _augment(self, obs, intended, effective, switched):
    if not self._log_self:
      return obs
    obs = dict(obs)
    obs['log/self_mode'] = np.float32(self._mode_index)
    obs['log/self_rho'] = np.float32(self._rho_default)
    obs['log/self_switch'] = np.float32(float(switched))
    obs['log/intended_action'] = np.float32(float(intended))
    obs['log/effective_action'] = np.float32(float(effective))
    obs['log/self_applied'] = np.float32(float(effective != intended))
    obs['log/self_mode_normal'] = np.float32(float(self._mode == 'normal'))
    obs['log/self_mode_sticky'] = np.float32(float(self._mode == 'sticky'))
    obs['log/self_mode_lazy'] = np.float32(float(self._mode == 'lazy'))
    obs['log/self_mode_confuse_lr'] = np.float32(
        float(self._mode == 'confuse_lr'))
    obs['log/self_mode_fire_dropout'] = np.float32(
        float(self._mode == 'fire_dropout'))
    return obs

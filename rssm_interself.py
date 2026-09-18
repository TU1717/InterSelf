"""InterSelf RSSM for the DreamerV3 JAX/Ninjax implementation.

This overlay is deliberately Atari-generic for discrete single-action games.
It models a hidden action channel as a row-stochastic confusion matrix:

  C_t[i, j] = P(a_effective=j | a_intended=i, self_state_t)
  p_map_t   = a_intended_t @ C_t
  p_eff_t   = (1-rho_t) * p_map_t + rho_t * p_eff_(t-1)

"""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import rssm as base_rssm

f32 = jnp.float32
sg = jax.lax.stop_gradient


class InterSelfRSSM(base_rssm.RSSM):
  """Official RSSM core with a categorical inferred action channel.

  The implementation supports a single discrete action component. The number
  of actions is inferred from ``act_space``; no game-specific index is
  hard-coded.
  
  """

  # Re-declared official RSSM fields. Do not remove: Ninjax ModuleMeta builds
  # this concrete class's field registry from __annotations__ only.
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  selfdim: int = 256
  selfhidden: int = 512
  selflayers: int = 2

  # Initial state is close to the identity channel and no sticky carry-over.
  alpha_init: float = -4.0
  rho_init: float = -4.0

  effect_layers: int = 1
  effect_units: int = 512
  selfreg_confusion: float = 1.0
  selfreg_alpha: float = 0.1
  selfreg_rho: float = 1.0

  # Capacity-matched control used in the paper. The execution module and its
  # auxiliary objectives remain active, but the world transition consumes the
  # intended action instead of the inferred effective action.
  identity_action: bool = False

  def __init__(self, act_space, **kw):
    super().__init__(act_space, **kw)
    self.actdim = self._infer_discrete_action_dim(act_space)

  def _infer_discrete_action_dim(self, act_space):
    # DreamerV3 make_agent removes reset, so Atari has one discrete `action`.
    if len(act_space) != 1:
      raise ValueError(
          'Categorical InterSelfRSSM currently supports one discrete action '
          f'component; received keys {tuple(act_space)}.')
    _, space = next(iter(act_space.items()))
    if not getattr(space, 'discrete', False):
      raise ValueError(
          'Categorical InterSelfRSSM is for discrete Atari actions; got '
          f'non-discrete action space {space}.')
    high = getattr(space, 'high', None)
    low = getattr(space, 'low', 0)
    if high is None:
      high = getattr(space, 'n', None)
    if high is None:
      raise ValueError(
          f'Could not infer the action cardinality from {space}.')
    dim = int(high) - int(low)
    if dim < 2:
      raise ValueError(f'Expected at least 2 discrete actions, got {dim}.')
    return dim

  @property
  def entry_space(self):
    spaces = dict(super().entry_space)
    spaces.update(dict(
        sdeter=elements.Space(np.float32, (self.selfdim,)),
        preveff=elements.Space(np.float32, (self.actdim,)),
    ))
    return spaces

  def _default_preveff(self, shape):
    # The bundled Atari wrapper uses action index 0 as NOOP (or its fallback),
    # so this mirrors the wrapper reset convention without using diagnostics.
    zeros = jnp.zeros(
        (*shape, self.actdim), dtype=nn.COMPUTE_DTYPE)
    return zeros.at[..., 0].set(1.0)

  def initial(self, bsize):
    carry = super().initial(bsize)
    carry.update(nn.cast(dict(
        sdeter=jnp.zeros([bsize, self.selfdim], f32),
        preveff=self._default_preveff((bsize,)),
    )))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    keys = ('deter', 'stoch', 'sdeter', 'preveff')
    return jax.tree.map(
        lambda x: x[:, -1], {key: entries[key] for key in keys})

  def starts(self, entries, carry, nlast):
    batch = len(jax.tree.leaves(carry)[0])
    keys = ('deter', 'stoch', 'sdeter', 'preveff')
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((batch * nlast, *x.shape[2:])),
        {key: entries[key] for key in keys})

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
    carry, (entries, feat) = nj.scan(
        lambda state, inputs: self._observe(state, *inputs, training),
        carry, (tokens, action, reset), unroll=unroll, axis=1)
    return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, sdeter, preveff = nn.mask(
        (carry['deter'], carry['stoch'], carry['sdeter'], carry['preveff']),
        ~reset)

    rawact = nn.DictConcat(self.act_space, 1)(action)
    rawact = nn.mask(rawact, ~reset)
    rawact = nn.cast(rawact)
    oldpreveff = jnp.where(
        reset[..., None], self._default_preveff(reset.shape), preveff)
    oldpreveff = nn.cast(oldpreveff)

    effact, operator = self._apply_categorical_operator(
        sdeter, stoch, rawact, oldpreveff)
    effact = nn.cast(effact)
    dynact = rawact if self.identity_action else effact
    deter = self._core(deter, stoch, nn.cast(dynact))

    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for index in range(self.obslayers):
      x = self.sub(f'obs{index}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(
          self.sub(f'obs{index}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    sdeter = self._self_core(
        sdeter, stoch, rawact, oldpreveff, effact)

    carry = dict(
        deter=deter,
        stoch=stoch,
        sdeter=sdeter,
        preveff=effact,
    )
    feat = dict(
        deter=deter,
        stoch=stoch,
        logit=logit,
        sdeter=sdeter,
        rawact=rawact,
        preveff=oldpreveff,
        effact=effact,
        self_confusion=operator['confusion'],
        self_alpha=operator['alpha'],
        self_rho=operator['rho'],
    )
    entry = dict(
        deter=deter,
        stoch=stoch,
        sdeter=sdeter,
        preveff=effact,
    )
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      rawact = nn.DictConcat(self.act_space, 1)(action)
      rawact = nn.cast(rawact)
      oldpreveff = nn.cast(carry['preveff'])
      effact, operator = self._apply_categorical_operator(
          carry['sdeter'], carry['stoch'], rawact, oldpreveff)
      effact = nn.cast(effact)
      dynact = rawact if self.identity_action else effact
      deter = self._core(
          carry['deter'], carry['stoch'], nn.cast(dynact))
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      sdeter = self._self_core(
          carry['sdeter'], stoch, rawact, oldpreveff, effact)
      carry = nn.cast(dict(
          deter=deter, stoch=stoch, sdeter=sdeter, preveff=effact))
      feat = nn.cast(dict(
          deter=deter,
          stoch=stoch,
          logit=logit,
          sdeter=sdeter,
          rawact=rawact,
          preveff=oldpreveff,
          effact=effact,
          self_confusion=operator['confusion'],
          self_alpha=operator['alpha'],
          self_rho=operator['rho'],
      ))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)

    unroll = length if self.unroll else 1
    if callable(policy):
      carry, (feat, action) = nj.scan(
          lambda state, _: self.imagine(
              state, policy, 1, training, single=True),
          nn.cast(carry), (), length, unroll=unroll, axis=1)
    else:
      carry, (feat, action) = nj.scan(
          lambda state, act: self.imagine(
              state, act, 1, training, single=True),
          nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
    return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    carry, entries, feat = self.observe(
        carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)

    losses = dict(
        dyn=dyn,
        rep=rep,
        selfeff=self._effect_loss(feat, tokens, reset),
        selfreg=self._self_regularization(feat),
    )

    confusion = feat['self_confusion']
    diagonal = jnp.diagonal(confusion, axis1=-2, axis2=-1)
    entropy = -jnp.sum(
        confusion * jnp.log(jnp.maximum(confusion, 1e-8)), axis=-1).mean()

    metrics = dict(
        dyn_ent=self._dist(prior).entropy().mean(),
        rep_ent=self._dist(post).entropy().mean(),
        **{
            'interself/rho_mean': feat['self_rho'].mean(),
            'interself/rho_std': feat['self_rho'].std(),
            'interself/alpha_mean': feat['self_alpha'].mean(),
            'interself/confusion_identity_mass': diagonal.mean(),
            'interself/confusion_offdiag_mass': 1.0 - diagonal.mean(),
            'interself/confusion_entropy': entropy,
            'interself/confusion_row_sum_error': (
                jnp.abs(confusion.sum(axis=-1) - 1.0).max()),
        },
    )
    return carry, entries, losses, feat, metrics

  def _self_core(self, sdeter, stoch, rawact, preveff, effact):
    sdeter, stoch, rawact, preveff, effact = nn.cast(
        (sdeter, stoch, rawact, preveff, effact))
    stoch = stoch.reshape((*stoch.shape[:-2], -1))
    x = jnp.concatenate([sdeter, stoch, rawact, preveff, effact], -1)
    for index in range(self.selflayers):
      x = self.sub(
          f'self{index}', nn.Linear, self.selfhidden, **self.kw)(x)
      x = nn.act(self.act)(
          self.sub(f'self{index}norm', nn.Norm, self.norm)(x))
    candidate = self.sub(
        'selfcand', nn.Linear, self.selfdim, outscale=0.1, **self.kw)(x)
    candidate = jnp.tanh(candidate)
    gate = self.sub(
        'selfgate', nn.Linear, self.selfdim, outscale=0.1, **self.kw)(x)
    gate = jax.nn.sigmoid(gate)
    return gate * candidate + (1.0 - gate) * sdeter

  def _operator_features(self, sdeter, stoch):
    sdeter, stoch = nn.cast((sdeter, stoch))
    stoch = stoch.reshape((*stoch.shape[:-2], -1))
    x = jnp.concatenate([sdeter, stoch], -1)
    for index in range(self.selflayers):
      x = self.sub(
          f'op{index}', nn.Linear, self.selfhidden, **self.kw)(x)
      x = nn.act(self.act)(
          self.sub(f'op{index}norm', nn.Norm, self.norm)(x))
    return x

  def _apply_categorical_operator(self, sdeter, stoch, rawact, preveff):
    # Match the RSSM mixed-precision convention before custom matrix
    # arithmetic.
    rawact, preveff = nn.cast((rawact, preveff))
    x = self._operator_features(sdeter, stoch)

    alpha_logits = self.sub(
        'op_alpha', nn.Linear, self.actdim, outscale=0.01, **self.kw)(x)
    alpha = jax.nn.sigmoid(alpha_logits + self.alpha_init)

    destination_logits = self.sub(
        'op_destination', nn.Linear, self.actdim * self.actdim,
        outscale=0.01, **self.kw)(x)
    destination_logits = destination_logits.reshape(
        (*x.shape[:-1], self.actdim, self.actdim))
    destination = jax.nn.softmax(destination_logits, axis=-1)

    identity = jnp.eye(self.actdim, dtype=nn.COMPUTE_DTYPE)
    confusion = (
        (1.0 - alpha[..., :, None]) * identity
        + alpha[..., :, None] * destination)
    confusion = nn.cast(confusion)

    mapped = jnp.einsum('...i,...ij->...j', rawact, confusion)
    mapped = nn.cast(mapped)

    rho_logits = self.sub(
        'op_rho', nn.Linear, 1, outscale=0.01, **self.kw)(x)
    rho = jax.nn.sigmoid(rho_logits + self.rho_init)[..., 0]

    effact = (
        (1.0 - rho[..., None]) * mapped
        + rho[..., None] * preveff)
    effact = nn.cast(effact)

    return effact, dict(confusion=confusion, alpha=alpha, rho=rho)

  def _effect_loss(self, feat, tokens, reset):
    token_dim = tokens.shape[-1]
    previous = jnp.concatenate(
        [jnp.zeros_like(tokens[:, :1]), tokens[:, :-1]], axis=1)
    target = sg(tokens - previous)
    mask = f32(~reset)
    deter, stoch, sdeter, effact = nn.cast((
        feat['deter'], feat['stoch'], feat['sdeter'], feat['effact']))
    stoch = stoch.reshape((*stoch.shape[:-2], -1))
    x = jnp.concatenate([deter, stoch, sdeter, effact], -1)
    for index in range(self.effect_layers):
      x = self.sub(
          f'effect{index}', nn.Linear, self.effect_units, **self.kw)(x)
      x = nn.act(self.act)(
          self.sub(f'effect{index}norm', nn.Norm, self.norm)(x))
    prediction = self.sub(
        'effect_out', nn.Linear, token_dim, outscale=0.0, **self.kw)(x)
    return ((prediction - target) ** 2).mean(-1) * mask

  def _self_regularization(self, feat):
    confusion = feat['self_confusion']
    identity = jnp.eye(self.actdim, dtype=confusion.dtype)
    confusion_loss = ((confusion - identity) ** 2).mean((-2, -1))
    alpha_loss = (feat['self_alpha'] ** 2).mean(-1)
    rho_loss = feat['self_rho'] ** 2
    return (
        self.selfreg_confusion * confusion_loss
        + self.selfreg_alpha * alpha_loss
        + self.selfreg_rho * rho_loss)

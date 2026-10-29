"""Explicit frozen baseline semantics; no learning or outcome access in deployment."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from scripts.cost_policy import _catalog, _finite
from scripts.pilot import PILOT, _unit
from scripts.policies import LinearUCB, ExploreOnly, EpochGreedy, RandomPolicy, FixedPolicy
from scripts.pretraining import _array, _digest, _integer, _read_json, _schema, _write_json

SUPPORTED = {p.__name__: p for p in (PILOT, LinearUCB, ExploreOnly, EpochGreedy, RandomPolicy, FixedPolicy)}
LINEAR = {'LinearUCB', 'ExploreOnly', 'EpochGreedy'}


def _validate_state(value):
    _schema(value, {'policy_type','arms','dimension','rounds','parameters','rng_state','A','b'},
            'frozen policy state')
    state = copy.deepcopy(value)
    kind = state['policy_type']
    if not isinstance(kind,str) or kind not in SUPPORTED:
        raise ValueError('Unsupported deployment policy')
    if not isinstance(state['arms'],list):
        raise ValueError('Snapshot arm catalog must be a list')
    arms = _catalog(state['arms'])
    dimension = _integer(state['dimension'],'dimension')
    _integer(state['rounds'],'rounds',0)
    parameters = state['parameters']
    if kind == 'PILOT':
        _schema(parameters, {'alpha','prior_provenance','priors','lambdas'}, 'PILOT parameters')
        _finite(parameters['alpha'],'alpha',minimum=0)
        if not isinstance(parameters['prior_provenance'],str) or not parameters['prior_provenance'].strip():
            raise ValueError('Prior provenance is required')
        _array(parameters['priors'],shape=(len(arms),dimension))
        if (_array(parameters['lambdas'],shape=(len(arms),)) <= 0).any():
            raise ValueError('Prior lambdas must be positive')
    elif kind in LINEAR:
        fields = {'alpha','ridge'} | ({'window'} if kind == 'EpochGreedy' else set())
        _schema(parameters, fields, 'ridge parameters')
        _finite(parameters['alpha'],'alpha',minimum=0)
        if _finite(parameters['ridge'],'ridge',minimum=0) == 0:
            raise ValueError('Ridge must be positive')
        if kind == 'EpochGreedy':
            _integer(parameters['window'],'window')
    elif kind == 'FixedPolicy':
        _schema(parameters, {'arm_id'}, 'fixed policy parameters')
        if parameters['arm_id'] not in arms:
            raise ValueError('Fixed arm is outside the model catalog')
    else:
        _schema(parameters,set(),'random policy parameters')
    try:
        probe = np.random.PCG64(0)
        probe.state = state['rng_state']
    except (TypeError,KeyError,ValueError,OverflowError) as exc:
        raise ValueError('Invalid learned RNG state') from exc
    theta = None
    if kind == 'PILOT' or kind in LINEAR:
        A = _array(state['A'],shape=(len(arms),dimension,dimension))
        b = _array(state['b'],shape=(len(arms),dimension))
        if not np.array_equal(A,A.swapaxes(-1,-2)):
            raise ValueError('Learned matrices must be symmetric')
        try:
            np.linalg.cholesky(A)
            with np.errstate(over='ignore',invalid='ignore'):
                theta = np.linalg.solve(A,b[...,None])[...,0]
        except np.linalg.LinAlgError as exc:
            raise ValueError('Learned matrices must be positive definite') from exc
        if not np.isfinite(theta).all():
            raise ValueError('Nonfinite learned predictions')
    elif state['A'] is not None or state['b'] is not None:
        raise ValueError('Nonlearning policy cannot contain ridge matrices')
    _digest(state)  # Also rejects non-JSON or nonfinite incidental fields.
    return state, theta


def policy_state(policy):
    """Copy learned state without calling its methods or consuming its RNG."""
    if type(policy) not in SUPPORTED.values():
        raise ValueError('Unsupported deployment policy; use an explicit adapter')
    kind = type(policy).__name__
    parameters = {}
    if kind == 'PILOT':
        parameters = dict(alpha=policy.alpha,prior_provenance=policy.provenance,
                          priors=policy._priors.tolist(),lambdas=policy._lambdas.tolist())
    elif kind in LINEAR:
        parameters = dict(alpha=policy.alpha,ridge=policy.ridge)
        if kind == 'EpochGreedy': parameters['window'] = policy.window
    elif kind == 'FixedPolicy': parameters = {'arm_id':policy.arm_id}
    state = dict(policy_type=kind,arms=list(policy.arms),dimension=policy.dimension,
                 rounds=policy.rounds,parameters=parameters,
                 rng_state=copy.deepcopy(policy.rng.bit_generator.state),
                 A=policy.A.tolist() if isinstance(policy,LinearUCB) else None,
                 b=policy.b.tolist() if isinstance(policy,LinearUCB) else None)
    return _validate_state(state)[0]


class FrozenDeploymentRouter:
    """Independent exploitation snapshot with a stateless per-query random order."""

    def __init__(self, policy, *, seed=42):
        self._install(policy_state(policy),seed)

    def _install(self,state,seed):
        self._seed = _integer(seed,'deployment seed',0)
        self._state, self._theta = _validate_state(state)
        self._kind = self._state['policy_type']
        self._arms = tuple(self._state['arms'])
        if self._kind == 'PILOT':
            self._theta = np.array([_unit(t,allow_zero=True) for t in self._theta])
        if self._theta is not None: self._theta.setflags(write=False)

    @property
    def arms(self): return self._arms

    @property
    def dimension(self): return self._state['dimension']

    @property
    def provenance(self):
        return self._state['parameters'].get('prior_provenance','baseline without preference prior')

    @property
    def score_domain(self): return 'linear' if self._kind in LINEAR else 'cosine'

    def source_state(self): return copy.deepcopy(self._state)

    def predict_rewards(self,context):
        x = _array(context,shape=(self.dimension,))
        if self._kind == 'PILOT':
            values = np.clip(self._theta @ _unit(x),-1.,1.)
        elif self._kind in LINEAR:
            with np.errstate(over='ignore',invalid='ignore'):
                values = self._theta @ x
        elif self._kind == 'RandomPolicy':
            values = np.full(len(self.arms),1./len(self.arms))
        else:
            values = np.array([float(a==self._state['parameters']['arm_id']) for a in self.arms])
        if not np.isfinite(values).all(): raise ValueError('Nonfinite frozen prediction')
        return dict(zip(self.arms,map(float,values)))

    def selection_options(self,query_id):
        if not isinstance(query_id,str) or not query_id.strip():
            raise ValueError('Query identity is required for reproducible selection')
        order = self.arms
        if self._kind == 'RandomPolicy':
            identity = _digest({'seed':self._seed,'query_id':query_id,'arms':list(self.arms)})
            rng = np.random.default_rng(int(identity,16))
            order = tuple(self.arms[int(i)] for i in rng.permutation(len(self.arms)))
        allowed = (self._state['parameters']['arm_id'],) if self._kind == 'FixedPolicy' else self.arms
        return dict(allowed_arms=allowed,tie_break_order=order)

    def metadata(self):
        return dict(version=1,policy_type=self._kind,source_state_sha256=_digest(self._state),
            source_model_frozen=True,deployment_seed=self._seed,score_domain=self.score_domain,
            score_semantics=('cosine' if self._kind=='PILOT' else 'unclipped_linear_ridge_mean'
                if self._kind in LINEAR else 'constant_1_over_K_utility' if self._kind=='RandomPolicy'
                else 'fixed_arm_indicator'),
            allowed_arms=list(self.selection_options('metadata')['allowed_arms']),
            tie_rule='seed_query_sha256_permutation' if self._kind=='RandomPolicy' else 'catalog_order',
            interpretation='LOCAL_DECLARED; see docs/common-deployment-protocol.md')

    def save_checkpoint(self,path):
        _write_json(Path(path),dict(version=1,kind='frozen-deployment-router',seed=self._seed,state=self._state))

    @classmethod
    def load_checkpoint(cls,path):
        payload = _read_json(Path(path))
        _schema(payload,{'version','kind','seed','state'},'frozen deployment checkpoint')
        if type(payload['version']) is not int or payload['version']!=1 or payload['kind']!='frozen-deployment-router':
            raise ValueError('Unsupported frozen deployment checkpoint')
        router = cls.__new__(cls)
        router._install(payload['state'],payload['seed'])
        return router

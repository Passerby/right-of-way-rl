from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PPODataFrame:
    actor_obs: Optional[Any] = None
    critic_obs: Optional[Any] = None
    next_obs: Optional[Any] = None
    myself_vector: Optional[Any] = None
    pre_action: Optional[Any] = None
    lstm_state: Optional[Any] = None
    reward: Optional[Any] = None
    action: Optional[Any] = None
    mask: Optional[Any] = None
    adv: Optional[Any] = None
    returns: Optional[Any] = None
    neg_logprob: Optional[Any] = None
    value: Optional[Any] = None
    done: Optional[Any] = None
    model_update_times: Optional[Any] = None

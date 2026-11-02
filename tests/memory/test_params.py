from memory.domain.entities import MemoryKind
from memory.domain.params import Params


def test_defaults_match_spec():
    p = Params.defaults()
    assert p.get("decay.episode") == 0.9 and p.get("recall_k") == 6 and p.get("extract_salience_threshold") == 1.5


def test_with_value_clamps_to_bounds():
    assert Params.defaults().with_value("recall_k", 99).get("recall_k") == 15


def test_invariants():
    assert Params.defaults().is_valid()
    assert not Params.defaults().with_value("dormant_threshold", 0.8).is_valid()
    assert not Params.defaults().with_value("memory_op_ask_threshold", 0.7).is_valid()


def test_decay_and_mode_prior_lookup():
    p = Params.defaults()
    assert p.decay(MemoryKind.FEATURE) == 0.2 and p.mode_prior("think") == 1.0 and p.mode_prior("code") == 0.5

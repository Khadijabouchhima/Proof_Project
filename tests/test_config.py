import copy
from dataclasses import replace
from types import MappingProxyType

import pytest

from biovance import config as C


def test_definitions_global_and_frozen():
    d = C.load_definitions()
    assert d.horizon_days == 30 and d.min_lead_days == 7
    assert d.reference.sbp == 130 and d.reference.maximum_gap_days == 7
    assert d.states == C.STATES
    with pytest.raises(TypeError):
        d.statistics["ci_level"] = 0.5


def test_global_file_has_no_simulator_or_real_specific_keys():
    raw = C._load_yaml(C.CONFIG_DIR / "definitions.yaml")
    assert not ({"baseline", "splits", "seeds", "reference_source"} & set(raw))
    assert "min_lead_days" not in raw["warning"]  # single authoritative value lives under task


def test_reference_rule_is_or_not_and():
    ref = C.load_definitions().reference
    assert ref.is_elevated(131, 70) and ref.is_elevated(120, 81) and not ref.is_elevated(129, 79)


def test_simulator_rules():
    r = C.load_simulator_rules()
    assert r.reference_source == "clinic_process" and r.baseline["window_days"] == 28
    assert r.seeds == (0, 1, 2, 3, 4)


def test_clean_t_ref_source_enforced():
    r = C.load_simulator_rules()
    bad = replace(r, reference_source="wearable_bp")
    with pytest.raises(ValueError, match="clinic_process"):
        C.validate_simulator_rules(bad)


def test_bad_splits_rejected():
    r = C.load_simulator_rules()
    bad = replace(r, splits=MappingProxyType({**r.splits, "train": 0.7}))
    with pytest.raises(ValueError, match="sum to 1"):
        C.validate_simulator_rules(bad)


def test_real_checks_forbid_labels_and_dryad_reference():
    c = C.load_real_checks()
    assert not c["common"]["uses_reference_event"]
    raw = C._load_yaml(C.CONFIG_DIR / "real_checks.yaml")
    bad = copy.deepcopy(raw)
    bad["common"]["uses_reference_event"] = True
    with pytest.raises(ValueError, match="no onset labels"):
        C.validate_real_checks(bad)
    bad = copy.deepcopy(raw)
    bad["datasets"]["dryad_24h"]["never_apply"] = []
    with pytest.raises(ValueError, match="dryad_24h"):
        C.validate_real_checks(bad)


def test_real_baselines_fit_their_datasets():
    c = C.load_real_checks()
    assert c["datasets"]["baigutanova_hrv"]["baseline_days"] <= 14  # recording is only 4 weeks
    assert c["datasets"]["dryad_24h"]["baseline_days"] is None


@pytest.mark.parametrize("name", ["sim_train", "sim_shifted"])
def test_sim_configs_valid(name):
    assert C.load_sim_config(name)["n_days"] == 180


def test_onset_inside_baseline_window_rejected():
    bad = copy.deepcopy(C._load_yaml(C.CONFIG_DIR / "sim_train.yaml"))
    bad["drift"]["onset_range"] = [20, 110]
    with pytest.raises(ValueError, match="contaminated"):
        C.validate_sim_config(bad)

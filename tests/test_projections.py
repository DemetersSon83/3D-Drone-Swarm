from __future__ import annotations

from drone_swarm.projections import AGENT_SIGNALS_CONTRACT_VERSION, template_agent_signal_row


def test_agent_signal_projection_is_stable_and_semantic() -> None:
    row = template_agent_signal_row()
    assert row["contract_version"] == AGENT_SIGNALS_CONTRACT_VERSION
    assert row["run_id"] == "template"
    assert row["observed_state_position_x"] == 0.0
    assert row["commanded_action_acceleration_x"] == 0.0
    assert row["true_state_position_x"] == 0.0
    assert row["applied_action_acceleration_x"] == 0.0
    assert row["true_next_state_position_x"] == 0.0
    assert len(row) == len(set(row))

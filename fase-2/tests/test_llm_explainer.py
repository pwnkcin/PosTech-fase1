from unittest.mock import MagicMock

import pytest

from src.llm_explainer import build_diagnosis_prompt, explain_diagnosis


def test_build_diagnosis_prompt_includes_patient_data_and_top_feature():
    prompt = build_diagnosis_prompt(
        biomarkers={"AMH": 8.2, "beta_HCG_I": 1.1, "beta_HCG_II": 0.5},
        prediction_label="Com PCOS",
        probability=0.87,
        top_shap_feature="AMH",
    )

    assert "8.2" in prompt
    assert "Com PCOS" in prompt
    assert "AMH" in prompt
    assert "0.87" in prompt or "87" in prompt


def test_explain_diagnosis_extracts_text_from_a_well_formed_client_response():
    mock_client = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Resumo: paciente apresenta risco elevado de PCOS."
    mock_client.messages.create.return_value.content = [mock_block]

    explanation = explain_diagnosis(
        client=mock_client,
        biomarkers={"AMH": 8.2, "beta_HCG_I": 1.1, "beta_HCG_II": 0.5},
        prediction_label="Com PCOS",
        probability=0.87,
        top_shap_feature="AMH",
    )

    assert explanation == "Resumo: paciente apresenta risco elevado de PCOS."


def test_explain_diagnosis_raises_a_clear_error_when_the_client_call_fails():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("connection refused")

    with pytest.raises(RuntimeError, match="Falha ao chamar a API da Anthropic"):
        explain_diagnosis(
            client=mock_client,
            biomarkers={"AMH": 8.2, "beta_HCG_I": 1.1, "beta_HCG_II": 0.5},
            prediction_label="Com PCOS",
            probability=0.87,
            top_shap_feature="AMH",
        )

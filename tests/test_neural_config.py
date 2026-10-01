from hre_translate.neural.config import load_nllb_config


def test_nllb_config_loads_required_training_controls() -> None:
    config, _ = load_nllb_config()
    assert config.model.name == "facebook/nllb-200-distilled-600M"
    assert config.model.source_language_proxy == "vie_Latn"
    assert config.training.batch_size > 0
    assert config.training.gradient_accumulation_steps > 0
    assert config.lora.rank == 8
    assert config.generation.beam_size == 4

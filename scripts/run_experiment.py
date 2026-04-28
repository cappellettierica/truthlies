from tlrs.config import ensure_output_dirs, load_config
from tlrs.data import DatasetLoader
from tlrs.experiment import ReasoningExperiment
from tlrs.models import CausalLanguageModel
from tlrs.utils import set_seed


def main() -> None:
    config = load_config("config.yaml")
    ensure_output_dirs(config)
    set_seed(config["project"]["seed"])

    loader = DatasetLoader(config)
    examples = loader.load_all(config)

    model = CausalLanguageModel(
        model_name=config["model"]["name"],
        max_new_tokens=config["model"]["max_new_tokens"],
        temperature=config["model"]["temperature"],
        do_sample=config["model"]["do_sample"],
        device=config["model"]["device"],
    )

    experiment = ReasoningExperiment(
        model=model,
        examples=examples,
        conditions=config["experiment"]["conditions"],
    )

    results = experiment.run()
    results.to_csv(config["outputs"]["results_file"], index=False)

    print(f"Saved results to {config['outputs']['results_file']}")


if __name__ == "__main__":
    main()
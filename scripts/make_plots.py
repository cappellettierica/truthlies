from tlrs.config import load_config
from tlrs.visualization import make_all_plots


def main() -> None:
    config = load_config("config.yaml")

    make_all_plots(
        results_path=config["outputs"]["results_file"],
        figures_dir=config["outputs"]["figures_dir"],
    )

    print(f"Saved figures to {config['outputs']['figures_dir']}")


if __name__ == "__main__":
    main()
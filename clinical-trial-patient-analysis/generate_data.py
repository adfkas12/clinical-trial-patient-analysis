"""Generate the project's synthetic clinical-trial dataset."""

from pathlib import Path

from src.config import DEFAULT_N_PATIENTS, DEFAULT_RANDOM_SEED, DATA_PATH
from src.data_generator import generate_synthetic_data, save_synthetic_data


def main() -> None:
    data = generate_synthetic_data(
        n_patients=DEFAULT_N_PATIENTS,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    output_path = save_synthetic_data(data, DATA_PATH)
    print(f"Generated {len(data)} synthetic participants: {output_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

Inputs = Literal["geostationary", "era5", "both"]
Mode = Literal["detection", "forecast", "forecast_oracle"]
CVScheme = Literal["holdout", "event_cv", "spatial_cv"]

# Channels of the climate_and_fuel cubes, in store order (Table 2 of the Pyrocast
# paper). u and v are the wind components at 250 hPa.
ERA5_VARIABLES = (
    "u10",
    "v10",
    "fg10",
    "blh",
    "cape",
    "cin",
    "z",
    "slhf",
    "sshf",
    "w",
    "u",
    "v",
    "cvh",
    "cvl",
    "tvh",
    "tvl",
    "r650",
    "r750",
    "r850",
)
Era5Variable = Literal[ERA5_VARIABLES]
EnvironmentVariable = Literal["longitude", "latitude", "date"]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Config):
    """Location and layout of the Pyrocast zarr stores.

    Attributes:
        root: Directory containing the three zarr stores and the snapshots CSV.
        geostationary_dir: Geostationary imagery store, relative to root.
        era5_dir: Climate and fuel store, relative to root.
        flags_dir: PyroCb flags and masks store, relative to root.
        snapshots_csv: Wildfire snapshots table, relative to root.
        events_csv: PyroCb events table (country, wildfire id), relative to root.
        index_cache: CSV file to cache the sample index in; rebuilt if missing.
        feature_cache: Pickle file to cache per-hour summary features in; missing
            hours are computed and appended. None computes them in memory.
        elevation_path: NetCDF elevation grid (lat, lon, data) used for the
            altitude variable of the ICP study. None leaves altitude out.
        countries: Only use pyroCb events from these countries (events_csv names,
            e.g. "Australia", "US", "Canada"). None uses all.
        max_events: Only use the first N events (sorted by id), for quick runs.
        n_jobs: Parallel jobs for reading data (joblib convention, -1 = all cores).
    """

    root: DirectoryPath
    geostationary_dir: str = "Geostationary_imagery"
    era5_dir: str = "climate_and_fuel"
    flags_dir: str = "PyroCb_flags_and_masks"
    snapshots_csv: str = "wildfire_snapshots.csv"
    events_csv: str = "pyrocb_events.csv"
    index_cache: Path | None = None
    feature_cache: Path | None = None
    elevation_path: FilePath | None = None
    countries: list[str] | None = Field(None, min_length=1)
    max_events: PositiveInt | None = None
    n_jobs: int = 8

    @field_validator("n_jobs")
    @classmethod
    def _check_n_jobs(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError("n_jobs must be a positive integer or -1")
        return v

    @model_validator(mode="after")
    def _check_stores_exist(self) -> "DataConfig":
        stores = [self.geostationary_path, self.era5_path, self.flags_path]
        missing = [str(p) for p in stores if not p.is_dir()]
        for table in (self.snapshots_path, self.events_path):
            if not table.is_file():
                missing.append(str(table))
        if missing:
            raise ValueError(f"Missing data paths: {', '.join(missing)}")
        return self

    @property
    def geostationary_path(self) -> Path:
        return self.root / self.geostationary_dir

    @property
    def era5_path(self) -> Path:
        return self.root / self.era5_dir

    @property
    def flags_path(self) -> Path:
        return self.root / self.flags_dir

    @property
    def snapshots_path(self) -> Path:
        return self.root / self.snapshots_csv

    @property
    def events_path(self) -> Path:
        return self.root / self.events_csv


class SplitConfig(_Config):
    """Train/test split or cross-validation, grouped by wildfire.

    All observations of a wildfire (every pyroCb and piece of it) share a split.

    Attributes:
        scheme: "holdout" holds out test_fraction of the wildfires once.
            "event_cv" assigns wildfires to n_folds folds at random, and
            "spatial_cv" by k-means on their latitude and longitude, as in the
            Pyrocast papers.
        test_fraction: Fraction of wildfires held out for "holdout".
        n_folds: Number of folds for the cross-validation schemes.
        seed: Random seed for the split or fold assignment.
    """

    scheme: CVScheme = "holdout"
    test_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    n_folds: int = Field(5, ge=2)
    seed: int = 0


class RandomForestConfig(_Config):
    """Random forest hyperparameters (defaults from the Pyrocast paper)."""

    n_estimators: PositiveInt = 500
    max_depth: PositiveInt | None = 10
    class_weight: Literal["balanced", "balanced_subsample"] | None = (
        "balanced_subsample"
    )
    random_state: int = 0
    n_jobs: int | None = None


class CNNConfig(_Config):
    """CNN training hyperparameters (defaults from the Pyrocast paper).

    Attributes:
        n_epochs: Classifier training epochs.
        pretrain_epochs: Auto-encoder pretraining epochs of the encoder; 0 trains
            the plain CNN, the paper's AE-CNN uses 40.
        batch_size: Mini-batch size.
        lr: Adam learning rate, for the classifier and the auto-encoder.
        num_workers: DataLoader worker processes.
        norm_samples: Training samples used to estimate channel mean and std.
        device: "auto" uses CUDA when available.
        seed: Seed for weight initialisation and shuffling.
    """

    n_epochs: PositiveInt = 5
    pretrain_epochs: int = Field(0, ge=0)
    batch_size: PositiveInt = 64
    lr: PositiveFloat = 0.001
    num_workers: int = Field(2, ge=0)
    norm_samples: PositiveInt = 512
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 0


class _ExperimentConfig(_Config):
    """Settings shared by all experiments.

    Attributes:
        mode: "detection" predicts the PyroCb flag at the input time, "forecast"
            the flag six hours ahead, and "forecast_oracle" the flag six hours
            ahead using ERA5 from the target time (a perfect weather forecast).
        era5_variables: ERA5 variables to use when inputs include ERA5, in any
            order; None uses all 19. The paper's "w3" set is cape, blh and r650.
    """

    experiment_name: str = Field(min_length=1)
    output_dir: Path
    mode: Mode = "forecast"
    inputs: Inputs = "geostationary"
    era5_variables: list[Era5Variable] | None = Field(None, min_length=1)
    data: DataConfig
    split: SplitConfig = SplitConfig()

    @model_validator(mode="after")
    def _check_oracle_inputs(self) -> "_ExperimentConfig":
        if self.mode == "forecast_oracle" and self.inputs == "geostationary":
            raise ValueError(
                "mode 'forecast_oracle' needs ERA5 inputs ('era5' or 'both'); "
                "with geostationary inputs alone it is the same as 'forecast'"
            )
        return self

    @model_validator(mode="after")
    def _check_era5_variables(self) -> "_ExperimentConfig":
        if self.era5_variables is None:
            return self
        if self.inputs == "geostationary":
            raise ValueError("era5_variables needs inputs 'era5' or 'both'")
        if len(set(self.era5_variables)) != len(self.era5_variables):
            raise ValueError("era5_variables has duplicates")
        return self

    @property
    def era5_channels(self) -> list[int]:
        """Store indices of the ERA5 variables to read, in store order."""
        names = self.era5_variables or ERA5_VARIABLES
        return sorted(ERA5_VARIABLES.index(n) for n in names)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_name


class RFExperimentConfig(_ExperimentConfig):
    """Random forest experiment: data, split, inputs and model."""

    model: RandomForestConfig = RandomForestConfig()


class CNNExperimentConfig(_ExperimentConfig):
    """CNN experiment: data, split, inputs and model."""

    model: CNNConfig = CNNConfig()


class ICPConfig(_Config):
    """Invariant causal prediction settings (defaults from the ICP paper).

    The conditional independence test Y _||_ E | X_S compares the cross-validated
    AUC of random forests on X_S and on (X_S, E) with DeLong's test.

    Attributes:
        n_estimators: Trees of the random forests inside the test.
        max_depth: Maximum tree depth.
        class_weight: Class weighting of the forests.
        random_state: Seed of the forests.
        n_jobs: Parallel tests (joblib convention, -1 = all cores).
        alpha: Significance level.
        environment: Environment variables E.
        exhaustive_variables: Variables for exhaustive ICP; None takes the last
            n_exhaustive variables of the greedy exclusion order.
        n_exhaustive: See exhaustive_variables.
        min_subset_size: Smallest subset tested by exhaustive ICP.
        hsic_threshold: Normalised HSIC at which two variables share a cluster.
        hsic_samples: Random observations used to estimate HSIC.
        validation_n_estimators: Trees of the forests that validate the greedy
            ordering with event and spatial cross-validation.
    """

    n_estimators: PositiveInt = 100
    max_depth: PositiveInt | None = 10
    class_weight: Literal["balanced", "balanced_subsample"] | None = (
        "balanced_subsample"
    )
    random_state: int = 0
    n_jobs: int = -1
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    environment: list[EnvironmentVariable] = Field(
        ["longitude", "latitude", "date"], min_length=1
    )
    exhaustive_variables: list[str] | None = Field(None, min_length=2)
    n_exhaustive: int = Field(11, ge=2)
    min_subset_size: PositiveInt = 8
    hsic_threshold: float = Field(0.25, gt=0.0, le=1.0)
    hsic_samples: int = Field(100, ge=10)
    validation_n_estimators: PositiveInt = 500


class ICPExperimentConfig(_ExperimentConfig):
    """ICP experiment: all geostationary and ERA5 variables, event CV for the test."""

    inputs: Inputs = "both"
    split: SplitConfig = SplitConfig(scheme="event_cv")
    model: ICPConfig = ICPConfig()

    @model_validator(mode="after")
    def _check_icp(self) -> "ICPExperimentConfig":
        if self.inputs != "both" or self.era5_variables is not None:
            raise ValueError("ICP uses all variables: inputs 'both', no era5_variables")
        if self.split.scheme == "holdout":
            raise ValueError("ICP needs a cross-validation split scheme")
        if self.data.elevation_path is None:
            raise ValueError("ICP needs data.elevation_path for the altitude variable")
        return self


C = TypeVar("C", bound=BaseModel)


def load_config(path: str | Path, config_cls: type[C]) -> C:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML file.
        config_cls: Config class to validate against.

    Returns:
        The validated config.
    """
    with open(path) as f:
        return config_cls.model_validate(yaml.safe_load(f))

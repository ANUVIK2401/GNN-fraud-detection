from dataclasses import dataclass, field, asdict
import json

# Columns that become entity nodes. Each transaction links to at most one
# entity per relation (its card1 value, its device, its email domain, ...).
# Continuous identity columns (id_01..id_11) from v1 are dropped: treating a
# float as a graph node adds noise, not structure.
DEFAULT_RELATIONS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "ProductCD",
    "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
]

# Low-cardinality categorical transaction columns that get one-hot encoded.
ONEHOT_COLS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

# Never used as model input features.
NON_FEATURE_COLS = ["TransactionID", "isFraud", "TransactionDT"]


@dataclass
class Config:
    relations: list = field(default_factory=lambda: list(DEFAULT_RELATIONS))
    train_frac: float = 0.6        # split is by TransactionDT, not random
    val_frac: float = 0.2
    k_neighbors: int = 10          # most recent prior txns per entity
    hidden: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 20
    patience: int = 3              # early stopping on val PR-AUC
    seed: int = 42

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

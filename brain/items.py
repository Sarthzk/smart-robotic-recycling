"""Waste items and the random spawner."""
from dataclasses import dataclass
import numpy as np
import config


@dataclass
class WasteItem:
    uid: int
    kind: str
    bin: str
    value: float
    contamination: float
    hazard: float
    cost: float
    arrival: float          # tick when the item enters the robot's reach
    color_id: int = 0       # "colour" of the item; used only by the colour baseline

    @property
    def recovered_value(self) -> float:
        """Money actually recovered if picked: dirty items are worth less."""
        return self.value * (1.0 - self.contamination)


def spawn_items(rng: np.random.Generator, n: int = config.N_ITEMS) -> list:
    kinds = list(config.ITEM_PROBS)
    probs = np.array([config.ITEM_PROBS[k] for k in kinds])
    probs = probs / probs.sum()
    gaps = rng.exponential(1.0 / config.ARRIVAL_RATE, size=n)
    arrivals = np.cumsum(gaps)
    items = []
    for i in range(n):
        kind = kinds[rng.choice(len(kinds), p=probs)]
        b, (vlo, vhi), hazard, cost, (clo, chi) = config.ITEM_TYPES[kind]
        items.append(WasteItem(
            uid=i, kind=kind, bin=b,
            value=float(rng.uniform(vlo, vhi)),
            contamination=float(rng.uniform(clo, chi)),
            hazard=hazard, cost=cost,
            arrival=float(arrivals[i]),
            # colours are random: colour tells you nothing about value
            color_id=int(rng.integers(0, 3)),
        ))
    return items

"""All tunable settings for the recycling simulation in one place."""
import numpy as np

# name: (bin, value range, hazard, recovery cost, contamination range)
#   value  = money recoverable from the item (higher is better)
#   hazard = 0..1, how dangerous it is if left on the belt
#   cost   = 0..3, how hard / slow it is to recover (heavy, awkward)
ITEM_TYPES = {
    "circuit_board": ("ewaste",    (7.0, 10.0), 0.10, 1.0, (0.0, 0.5)),
    "phone":         ("ewaste",    (6.0, 9.0),  0.20, 1.0, (0.0, 0.4)),
    "copper_wire":   ("metal",     (4.0, 6.0),  0.00, 1.5, (0.0, 0.6)),
    "aluminium_can": ("metal",     (1.5, 2.5),  0.00, 0.5, (0.0, 0.7)),
    "steel_scrap":   ("metal",     (1.0, 2.0),  0.05, 3.0, (0.0, 0.5)),
    "pet_bottle":    ("plastic",   (0.8, 1.5),  0.00, 0.5, (0.0, 0.8)),
    "hdpe_plastic":  ("plastic",   (0.5, 1.2),  0.00, 1.0, (0.0, 0.8)),
    "li_ion_battery": ("hazardous", (3.0, 5.0), 0.70, 1.0, (0.0, 0.3)),
    "leaking_battery": ("hazardous", (1.0, 2.0), 1.00, 1.5, (0.3, 0.9)),
}
# how often each type appears on the belt
ITEM_PROBS = {
    "circuit_board": 0.08, "phone": 0.08, "copper_wire": 0.10,
    "aluminium_can": 0.18, "steel_scrap": 0.10, "pet_bottle": 0.20,
    "hdpe_plastic": 0.14, "li_ion_battery": 0.07, "leaking_battery": 0.05,
}

BINS = ["ewaste", "metal", "plastic", "hazardous", "overflow"]
BIN_CAPACITY = {"ewaste": 12, "metal": 12, "plastic": 12, "hazardous": 8, "overflow": 10}
OVERFLOW_VALUE_FACTOR = 0.5       # items sent to the overflow bin are worth less (mixed)

# episode / belt settings
N_ITEMS = 40                      # items per episode
ARRIVAL_RATE = 0.9                # average items arriving per tick (robot cannot keep up)
DWELL = 7                         # ticks an item stays inside the robot's reach
PICK_TIME = 1.0                   # base ticks for a pick; plus cost * COST_TIME
COST_TIME = 0.6
TIME_PENALTY = 0.05               # reward lost per tick of robot time

# gripper failure / backup tool
GRIPPER_FAIL_PROB = 0.10          # scaled up for hard (high cost) items
MAX_GRIPPER_FAILS = 2             # consecutive fails before switching to backup tool
BACKUP_TOOL_SLOWDOWN = 1.5        # backup suction tool is reliable but slower

# reward shaping
HAZARD_PICK_BONUS = 4.0           # extra reward for removing a hazardous item (x hazard)
HAZARD_ESCAPE_PENALTY = 8.0       # penalty if a hazardous item escapes (x hazard)

# scoring features (see brain/scoring.py)
FEATURES = ["value", "contamination", "hazard", "cost", "urgency"]
# naive hand-made weights: this is the "fixed score" baseline the robot should beat
DEFAULT_WEIGHTS = np.array([1.0, -1.0, 1.0, -1.0, 0.0])

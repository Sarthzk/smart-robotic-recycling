"""PyBullet scene: floor, conveyor belt, five bins and the waste items.

Belt motion and carried items are moved kinematically from the simulation clock
(so the belt always matches the logic in brain/env.py); the robot arm itself runs on PyBullet physics."""
import math
import numpy as np
import pybullet as p
import pybullet_data
import config

BELT_Y = 0.50
BELT_TOP = 0.06
REACH_X = 0.42                 # belt x where an item enters / leaves the robot's reach
BELT_END = 0.66
SPEED = 2 * REACH_X / config.DWELL            # metres per tick
BIN_Y = -0.50
BIN_X = {"ewaste": -0.50, "metal": -0.25, "plastic": 0.0, "hazardous": 0.25, "overflow": 0.50}
BIN_RGBA = {"ewaste": (0.16, 0.42, 0.85, 1), "metal": (0.58, 0.64, 0.70, 1), "plastic": (0.96, 0.76, 0.18, 1),
            "hazardous": (0.86, 0.17, 0.17, 1), "overflow": (0.55, 0.36, 0.80, 1)}
BIN_TITLE = {"ewaste": "E-WASTE", "metal": "METAL", "plastic": "PLASTIC", "hazardous": "HAZARDOUS", "overflow": "OVERFLOW"}
CAM = dict(target=[0.0, 0.12, 0.08], dist=1.5, yaw=32, pitch=-40)
FLOOR_RGBA = (0.80, 0.82, 0.85, 1)          # light factory floor
MAT_RGBA = (0.17, 0.19, 0.22, 1)            # dark work-cell mat
SAFETY_RGBA = (0.98, 0.78, 0.10, 1)         # yellow safety lines
ITEM_SCALE = 1.3                            # items are drawn a bit larger so they are easy to see
# every item uses one of these dull colours picked at random: colour tells you nothing about value
ITEM_RGBA = [(0.45, 0.45, 0.47, 1), (0.40, 0.30, 0.22, 1), (0.25, 0.38, 0.30, 1)]
SHAPES = {"circuit_board": ("box", (0.035, 0.025, 0.006)), "phone": ("box", (0.035, 0.018, 0.006)),
          "copper_wire": ("cyl", (0.02, 0.05)), "aluminium_can": ("cyl", (0.02, 0.06)),
          "steel_scrap": ("box", (0.03, 0.03, 0.02)), "pet_bottle": ("cyl", (0.02, 0.07)),
          "hdpe_plastic": ("box", (0.03, 0.025, 0.02)), "li_ion_battery": ("box", (0.025, 0.015, 0.015)),
          "leaking_battery": ("box", (0.025, 0.015, 0.015))}


def belt_x(item, vt):
    return REACH_X - SPEED * (vt - item.arrival)


class World:
    def __init__(self, items, headless=False):
        self.headless = headless
        self.cid = p.connect(p.DIRECT if headless else p.GUI)
        if not headless:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)      # shadows are slow in the macOS window
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        plane = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane, -1, textureUniqueId=-1, rgbaColor=FLOOR_RGBA)    # plain floor, no checkerboard
        self.bin_text, self.bin_last = {}, {}
        self._build_cell()
        self._build_belt()
        self._build_bins()
        self.items = items
        self.body, self.label, self.shape_half = {}, {}, {}
        for it in items:
            self._make_item(it)
        self.in_bin = {b: 0 for b in config.BINS}
        self.status_text = -1
        self.label_rank = None          # {uid: rank} -> label only these items (None = label all)
        self.label_text = {}
        p.resetDebugVisualizerCamera(CAM['dist'], CAM['yaw'], CAM['pitch'], CAM['target'])

    # ------------------------------------------------------------ static scene
    def _box(self, half, pos, rgba, mass=0):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        return p.createMultiBody(mass, col, vis, pos)

    def _build_cell(self):
        """Dark mat with yellow safety lines, a back wall with the project title and a robot base plate."""
        self._box([1.05, 0.95, 0.002], [0, 0.0, 0.002], MAT_RGBA)
        for (hx, hy, x, y) in ((1.05, 0.012, 0, 0.945), (1.05, 0.012, 0, -0.945),
                               (0.012, 0.95, 1.045, 0), (0.012, 0.95, -1.045, 0)):
            self._box([hx, hy, 0.003], [x, y, 0.0045], SAFETY_RGBA)
        self._box([3.0, 0.02, 0.75], [0, 2.1, 0.75], (0.17, 0.23, 0.31, 1))                  # back wall, far enough that text never touches it
        self._box([3.0, 0.022, 0.03], [0, 2.09, 0.12], SAFETY_RGBA)                           # wall stripe
        self._text("SMART ROBOTIC RECYCLING SYSTEM", [-0.95, 0.80, 0.80], (1, 1, 1), 1.6)
        base = p.createVisualShape(p.GEOM_CYLINDER, radius=0.17, length=0.02, rgbaColor=(0.30, 0.32, 0.36, 1))
        p.createMultiBody(0, -1, base, [0, 0, 0.012])

    def _build_belt(self):
        L = BELT_END + 0.05
        self._box([L, 0.13, 0.012], [0, BELT_Y, BELT_TOP - 0.012], (0.10, 0.10, 0.11, 1))       # belt surface
        for sy in (-1, 1):                                                                       # steel side rails
            self._box([L, 0.012, 0.028], [0, BELT_Y + sy * 0.142, BELT_TOP - 0.01], (0.72, 0.75, 0.79, 1))
        for sx in (-1, 1):                                                                       # end rollers
            r = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.27, rgbaColor=(0.25, 0.26, 0.28, 1))
            p.createMultiBody(0, -1, r, [sx * (L - 0.005), BELT_Y, BELT_TOP - 0.03],
                              p.getQuaternionFromEuler([math.pi / 2, 0, 0]))
        for lx in (-0.6, -0.2, 0.2, 0.6):                                                        # legs
            for sy in (-1, 1):
                self._box([0.014, 0.014, 0.0225], [lx, BELT_Y + sy * 0.142, 0.0225], (0.55, 0.58, 0.62, 1))
        self.stripes = [self._box([0.006, 0.115, 0.001], [0, BELT_Y, BELT_TOP + 0.001], (0.30, 0.31, 0.34, 1))
                        for _ in range(14)]
        self.stripe_gap = 2 * BELT_END / len(self.stripes)

    def _build_bins(self):
        for name, x in BIN_X.items():
            c = BIN_RGBA[name]
            dark = (c[0] * 0.55, c[1] * 0.55, c[2] * 0.55, 1)
            self._box([0.10, 0.10, 0.004], [x, BIN_Y, 0.006], dark)
            for dx, dy, hx, hy in ((0.10, 0, 0.004, 0.10), (-0.10, 0, 0.004, 0.10),
                                   (0, 0.10, 0.10, 0.004), (0, -0.10, 0.10, 0.004)):
                self._box([hx, hy, 0.04], [x + dx, BIN_Y + dy, 0.044], c)
            self._box([0.10, 0.006, 0.012], [x, BIN_Y - 0.106, 0.012], c)                        # colour plate on the floor
            self.bin_text[name] = -1
        self.update_bin_counters({b: 0 for b in BIN_X}, config.BIN_CAPACITY)

    def update_bin_counters(self, counts, capacity):
        """Text above each bin: NAME n/cap, turns red when full."""
        for name, x in BIN_X.items():
            n, cap = counts.get(name, 0), capacity[name]
            state = (n, cap)
            if self.bin_last.get(name) == state:
                continue
            self.bin_last[name] = state
            full = n >= cap
            txt = f"{BIN_TITLE[name]} {n}/{cap}" + ("  FULL" if full else "")
            rgb = (1.0, 0.35, 0.35) if full else (1, 1, 1)
            pos = [x - 0.085, BIN_Y - 0.15, 0.13]
            if self.bin_text.get(name, -1) != -1:
                self.bin_text[name] = self._text(txt, pos, rgb, 1.0, replaceItemUniqueId=self.bin_text[name])
            else:
                self.bin_text[name] = self._text(txt, pos, rgb, 1.0)

    def _text(self, txt, pos, rgb, size, **kw):
        try:
            return p.addUserDebugText(txt, pos, rgb, size, **kw)
        except Exception:
            return -1

    # ------------------------------------------------------------ items
    def _make_item(self, it):
        kind, dims = SHAPES[it.kind]
        dims = tuple(d * ITEM_SCALE for d in dims)
        rgba = ITEM_RGBA[it.color_id]
        if kind == "box":
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dims, rgbaColor=rgba)
            half_h = dims[2]
        else:
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=dims[0], length=dims[1], rgbaColor=rgba)
            half_h = dims[0]
        # cylinders lie on their side
        orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0]) if kind == "cyl" else [0, 0, 0, 1]
        body = p.createMultiBody(0, -1, vis, [0, 0, -5], orn)
        self.body[it.uid], self.shape_half[it.uid] = body, half_h
        self.label[it.uid] = -1            # labels are created only while the item is on the belt (speed)

    def _show_label(self, it, on, text=None):
        if self.headless:
            return
        has = self.label[it.uid] != -1
        if has and (not on or self.label_text.get(it.uid) != text):
            try:
                p.removeUserDebugItem(self.label[it.uid])
            except Exception:
                pass
            self.label[it.uid] = -1
            has = False
        if on and not has:
            z = self.shape_half[it.uid] + 0.03 + 0.03 * (it.uid % 3)       # stagger so neighbours do not overlap
            self.label[it.uid] = self._text(text, [0, 0, z], (1, 1, 1), 0.9, parentObjectUniqueId=self.body[it.uid])
            self.label_text[it.uid] = text

    def _label_for(self, it):
        """Text to show above an item, or None for no label (only the top-ranked items are labelled)."""
        if self.label_rank is None:
            return f"{it.kind} ${it.recovered_value:.1f}"
        rank = self.label_rank.get(it.uid)
        return None if rank is None else f"#{rank} {it.kind} ${it.recovered_value:.1f}"

    def item_pos_on_belt(self, it, vt):
        return [belt_x(it, vt), BELT_Y, BELT_TOP + self.shape_half[it.uid]]

    def place_item(self, it, pos):
        _, orn = p.getBasePositionAndOrientation(self.body[it.uid])
        p.resetBasePositionAndOrientation(self.body[it.uid], pos, orn)

    def drop_in_bin(self, it, dest):
        n = self.in_bin[dest]
        self.in_bin[dest] += 1
        col, row, layer = n % 3, (n // 3) % 3, n // 9
        x = BIN_X[dest] + (col - 1) * 0.055
        y = BIN_Y + (row - 1) * 0.055
        self.place_item(it, [x, y, 0.012 + self.shape_half[it.uid] + layer * 0.03])
        self._show_label(it, False)

    def update_belt(self, vt, picked):
        """Place every item and stripe for visual time vt. Picked items are handled elsewhere."""
        for it in self.items:
            if it.uid in picked:
                continue
            x = belt_x(it, vt)
            if -BELT_END <= x <= BELT_END:
                self.place_item(it, self.item_pos_on_belt(it, vt))
                text = self._label_for(it)
                self._show_label(it, text is not None, text)
            else:
                self.place_item(it, [0, 0, -5])          # not on the belt yet / drove off the end
                self._show_label(it, False)
        shift = (SPEED * vt) % self.stripe_gap
        for i, s in enumerate(self.stripes):
            x = BELT_END - ((i * self.stripe_gap + shift) % (2 * BELT_END))
            p.resetBasePositionAndOrientation(s, [x, BELT_Y, BELT_TOP + 0.002], [0, 0, 0, 1])

    def set_status(self, text):
        if self.headless:
            return
        pos, size = [-0.95, 0.80, 0.66], 1.1            # under the title, well in front of the wall
        self.status_text = self._text(text, pos, (1, 1, 0.4), size, replaceItemUniqueId=self.status_text) \
            if self.status_text != -1 else self._text(text, pos, (1, 1, 0.4), size)

    def screenshot(self, width=960, height=600):
        view = p.computeViewMatrixFromYawPitchRoll(CAM['target'], CAM['dist'], CAM['yaw'], CAM['pitch'], 0, 2)
        proj = p.computeProjectionMatrixFOV(60, width / height, 0.05, 5)
        img = p.getCameraImage(width, height, view, proj)[2]
        return np.reshape(img, (height, width, 4))[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect(self.cid)

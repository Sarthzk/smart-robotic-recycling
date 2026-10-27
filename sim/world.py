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
BIN_RGBA = {"ewaste": (0.2, 0.4, 0.9, 1), "metal": (0.6, 0.65, 0.7, 1), "plastic": (0.95, 0.8, 0.2, 1),
            "hazardous": (0.9, 0.15, 0.15, 1), "overflow": (0.7, 0.4, 0.9, 1)}
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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.loadURDF("plane.urdf")
        self._build_belt()
        self._build_bins()
        self.items = items
        self.body, self.label, self.shape_half = {}, {}, {}
        for it in items:
            self._make_item(it)
        self.in_bin = {b: 0 for b in config.BINS}
        self.status_text = -1
        p.resetDebugVisualizerCamera(1.55, 28, -48, [0.0, 0.0, 0.05])

    # ------------------------------------------------------------ static scene
    def _box(self, half, pos, rgba, mass=0):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        return p.createMultiBody(mass, col, vis, pos)

    def _build_belt(self):
        self._box([BELT_END + 0.05, 0.13, BELT_TOP / 2], [0, BELT_Y, BELT_TOP / 2], (0.12, 0.12, 0.13, 1))
        for sy in (-1, 1):                      # side rails
            self._box([BELT_END + 0.05, 0.01, 0.03], [0, BELT_Y + sy * 0.14, BELT_TOP + 0.01], (0.85, 0.7, 0.1, 1))
        self.stripes = [self._box([0.006, 0.12, 0.002], [0, BELT_Y, BELT_TOP + 0.002], (0.25, 0.25, 0.27, 1))
                        for _ in range(14)]
        self.stripe_gap = 2 * BELT_END / len(self.stripes)

    def _build_bins(self):
        for name, x in BIN_X.items():
            c = BIN_RGBA[name]
            self._box([0.10, 0.10, 0.004], [x, BIN_Y, 0.004], c)
            for dx, dy, hx, hy in ((0.10, 0, 0.004, 0.10), (-0.10, 0, 0.004, 0.10),
                                   (0, 0.10, 0.10, 0.004), (0, -0.10, 0.10, 0.004)):
                self._box([hx, hy, 0.04], [x + dx, BIN_Y + dy, 0.04], c)
            self._text(name.upper(), [x - 0.07, BIN_Y - 0.15, 0.01], (1, 1, 1), 1.0)

    def _text(self, txt, pos, rgb, size, **kw):
        try:
            return p.addUserDebugText(txt, pos, rgb, size, **kw)
        except Exception:
            return -1

    # ------------------------------------------------------------ items
    def _make_item(self, it):
        kind, dims = SHAPES[it.kind]
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
        self.label[it.uid] = self._text(f"{it.kind} ${it.recovered_value:.1f}", [0, 0, half_h + 0.03],
                                        (1, 1, 1), 0.9, parentObjectUniqueId=body)

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
        try:
            p.removeUserDebugItem(self.label[it.uid])
        except Exception:
            pass

    def update_belt(self, vt, picked):
        """Place every item and stripe for visual time vt. Picked items are handled elsewhere."""
        for it in self.items:
            if it.uid in picked:
                continue
            x = belt_x(it, vt)
            if -BELT_END <= x <= BELT_END:
                self.place_item(it, self.item_pos_on_belt(it, vt))
            else:
                self.place_item(it, [0, 0, -5])          # not on the belt yet / drove off the end
        shift = (SPEED * vt) % self.stripe_gap
        for i, s in enumerate(self.stripes):
            x = BELT_END - ((i * self.stripe_gap + shift) % (2 * BELT_END))
            p.resetBasePositionAndOrientation(s, [x, BELT_Y, BELT_TOP + 0.002], [0, 0, 0, 1])

    def set_status(self, text):
        self.status_text = self._text(text, [-0.62, -0.1, 0.62], (1, 1, 1), 1.3,
                                      replaceItemUniqueId=self.status_text) if self.status_text != -1 \
            else self._text(text, [-0.62, -0.1, 0.62], (1, 1, 1), 1.3)

    def screenshot(self, width=960, height=600):
        view = p.computeViewMatrixFromYawPitchRoll([0.0, 0.0, 0.05], 1.55, 28, -48, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, width / height, 0.05, 5)
        img = p.getCameraImage(width, height, view, proj)[2]
        return np.reshape(img, (height, width, 4))[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect(self.cid)

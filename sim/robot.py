"""KUKA iiwa arm with a simple tool. Moves with PyBullet inverse kinematics + position control."""
import math
import numpy as np
import pybullet as p

EE = 6                       # iiwa end-effector link
TIP = 0.12                   # distance from flange to tool tip
DOWN = p.getQuaternionFromEuler([math.pi, 0, 0])
HOME_TIP = np.array([0.0, 0.28, 0.38])
SAG = 0.02                   # PyBullet's arm sags ~2 cm under gravity; aim a bit higher
LOWER = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
UPPER = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]
RANGES = [5.8, 4.0, 5.8, 4.0, 5.8, 4.0, 6.0]
REST = [0, 0.4, 0, -1.6, 0, 1.2, 0]
TOOL_RGBA = {"gripper": (0.1, 0.8, 0.3, 1), "fail": (1.0, 0.1, 0.1, 1), "suction": (1.0, 0.7, 0.0, 1)}


class Robot:
    def __init__(self):
        self.id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.02], useFixedBase=True)
        for j, q in enumerate(REST):
            p.resetJointState(self.id, j, q)
        dark = (0.15, 0.16, 0.18, 1)
        def vbox(half, rgba):
            return p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba), [0, 0, 1])
        self.palm = vbox([0.03, 0.03, 0.015], TOOL_RGBA["gripper"])
        self.finger_l = vbox([0.006, 0.02, 0.03], dark)
        self.finger_r = vbox([0.006, 0.02, 0.03], dark)
        self.cup = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.03,
                                                                rgbaColor=TOOL_RGBA["suction"]), [0, 0, 1])
        self.key = "gripper"
        self.closed = False
        self.tip_target = HOME_TIP.copy()

    def tip_position(self):
        pos, orn = p.getLinkState(self.id, EE)[:2]
        return np.array(pos) + np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)[:, 2] * TIP

    def set_tool_color(self, key):
        """key: 'gripper' (normal), 'fail' (red) or 'suction' (backup tool)."""
        self.key = key
        p.changeVisualShape(self.palm, -1, rgbaColor=TOOL_RGBA[key])

    def set_grip(self, closed):
        self.closed = closed

    def command(self, tip):
        """Drive the arm so the tool tip heads to `tip` (world xyz)."""
        self.tip_target = np.array(tip, dtype=float)
        flange = self.tip_target + np.array([0, 0, TIP + SAG])
        q = p.calculateInverseKinematics(self.id, EE, flange.tolist(), DOWN, LOWER, UPPER, RANGES, REST,
                                         maxNumIterations=60)
        p.setJointMotorControlArray(self.id, list(range(7)), p.POSITION_CONTROL, targetPositions=q[:7],
                                    forces=[300] * 7, positionGains=[0.25] * 7, velocityGains=[1.0] * 7)

    def update_tool_visual(self):
        tip = self.tip_position()
        x, y, z = tip
        p.resetBasePositionAndOrientation(self.palm, [x, y, z + 0.075], [0, 0, 0, 1])
        if self.key == "suction":                       # backup tool: one suction cup, no fingers
            p.resetBasePositionAndOrientation(self.cup, [x, y, z + 0.03], [0, 0, 0, 1])
            for f in (self.finger_l, self.finger_r):
                p.resetBasePositionAndOrientation(f, [x, y, -5], [0, 0, 0, 1])
        else:
            gap = 0.017 if self.closed else 0.034
            p.resetBasePositionAndOrientation(self.cup, [x, y, -5], [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(self.finger_l, [x - gap, y, z + 0.03], [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(self.finger_r, [x + gap, y, z + 0.03], [0, 0, 0, 1])
        return tip

    def settle(self, tip, steps=240):
        self.command(tip)
        for _ in range(steps):
            p.stepSimulation()
        self.update_tool_visual()

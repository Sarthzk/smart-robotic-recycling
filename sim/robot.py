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
        self.id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        for j, q in enumerate(REST):
            p.resetJointState(self.id, j, q)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.04], rgbaColor=TOOL_RGBA["gripper"])
        self.tool = p.createMultiBody(0, -1, vis, [0, 0, 1])
        self.mode = "gripper"
        self.tip_target = HOME_TIP.copy()

    def tip_position(self):
        pos, orn = p.getLinkState(self.id, EE)[:2]
        return np.array(pos) + np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)[:, 2] * TIP

    def set_tool_color(self, key):
        p.changeVisualShape(self.tool, -1, rgbaColor=TOOL_RGBA[key])

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
        p.resetBasePositionAndOrientation(self.tool, (tip + [0, 0, 0.04]).tolist(), [0, 0, 0, 1])
        return tip

    def settle(self, tip, steps=240):
        self.command(tip)
        for _ in range(steps):
            p.stepSimulation()
        self.update_tool_visual()

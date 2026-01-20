from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.gripper.xense.xense import Xense

robot = Rizon(id='Rizon4s-063231', gripper=False, name='Rizon4s')
robot.homing()

gripper = Xense(id='5e77ff097831', name='Xense')
gripper.move(0.08)

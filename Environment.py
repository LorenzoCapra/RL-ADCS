import gym
import numpy as np
import random
from numba import jit
from gym import spaces, logger
from scipy.integrate import solve_ivp
from TOOLS.Dynamics import dynamics


class AttitudeControlEnv(gym.Env):

    # ---- toolbox ----
    # all the typical functions needed for quaternion math stuff.
    # set as staticmethod so numba will compile them for speeds

    @staticmethod
    @jit(nopython=True)
    def _randomAxisAngle(min_angle, max_angle):
        """
        generates random axis angle slew in form of
        (x, y, z, angle)
        angle in radians
        """
        # generate phi and theta. then convert from spherical --> cartesian
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(-1, 1)

        theta = np.arccos(costheta)  # takes care of quadrant issue
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # now snag our angle for the slew from the function bounds
        angle = np.random.uniform(min_angle, max_angle) * np.pi / 180

        return np.array([x, y, z, angle])

    @staticmethod
    @jit(nopython=True)
    def _axisAngleToQuat(axis_angle):
        """
        requires np array of axis angle in [x, y, z, angle].
        angle in rad. returns quaternion in [qv q4]
        where q4 is real part and qv = q1 q2 q3
        """
        ehat_x, ehat_y, ehat_z, ang = axis_angle

        q1 = ehat_x * np.sin(ang / 2)
        q2 = ehat_y * np.sin(ang / 2)
        q3 = ehat_z * np.sin(ang / 2)
        q4 = np.cos(ang / 2)

        return np.array([q1, q2, q3, q4])

    # -------end toolbox, start actual env-------

    def __init__(self, state0, steps=500, precision=0.25, test=False):

        self.state0 = state0
        self.qs_prev = state0[3]

        # If you are testing set to True
        self.test = test

        # Number of times to integrate between actions
        self.frameskip_num = 20  # controls agent fidelity

        # Desired angle
        self.precision = precision

        self.max_angle_slew = 150.  # maximum angle for goal slew generation, in degrees
        self.min_angle_slew = 30.  # minimum angle for goal slew generation, in degrees
        self.initial_angle = 0.

        self.I = np.array([0.872, 0.115, 0.797])  # rotational inertia tensor (assuming symmetric, this is the diagonal)

        self.h = 1 / 240  # integrator fidelity (in seconds)

        # Initial quaternion (in world frame--always starts aligned with world frame)
        self.q_initial = np.array([0., 0., 0., 1.])

        high = np.array([1.0, 1.0, 1.0, 1.0, 10, 10, 10, 10, 1.0, 1.0, 1.0])
        self.action_space = spaces.Box(-1, +1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.nsteps = None
        self.steps_beyond_done = None

        # Multiplying factor for the reward when close to desired angle
        self.alpha = np.array([-1., -1., -1., 1.])

        # ---thresholds for episode-----

        self.max_episode_steps = steps
        self.max_ang_velo = 0.5  # [rad/s]

        # --------------------------

    def step(self, action):

        self.nsteps += 1

        # Add instantaneous disturbance torque
        # if self.nsteps == 200:
        #    action += [5, 2, 1]

        # Step the env once with the torque applied:
        sol = solve_ivp(dynamics, [0, self.h], self.state0,
                        method='LSODA', t_eval=[self.h],
                        args=(self.I, action))
        # Update the state
        self.state0 = np.squeeze(sol.y)

        # empty the torques
        torque = np.array([0., 0., 0.])
        # Add continuous disturbance torque
        # torque += np.array([random.uniform(1e-5, 1e-3), random.uniform(1e-5, 1e-3), random.uniform(1e-5, 1e-3)])
        # torque *= np.array([random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])])

        # propagate free rotation forward
        for _ in range(self.frameskip_num):
            sol = solve_ivp(dynamics, [0, self.h], self.state0,
                            method='LSODA', t_eval=[self.h],
                            args=(self.I, torque))
            # Update the state
            self.state0 = np.squeeze(sol.y)

        qs = self.state0[3]
        curr_angle = 2 * np.arccos(qs)
        omega_magnitude = np.linalg.norm(self.state0[8:])

        done = omega_magnitude > self.max_ang_velo or self.nsteps >= self.max_episode_steps
        done = bool(done)
        exceed = bool(omega_magnitude > self.max_ang_velo)

        # -------- REWARD --------- #
        if not done:
            arg = curr_angle / (0.14 * 2 * np.pi)
            if qs > self.qs_prev:
                reward = np.exp(-arg)
            else:
                reward = np.exp(-arg) - 1.

            # reward = - abs(self.state0[0]) - abs(self.state0[1]) - abs(self.state0[2]) + abs(self.state0[3])

            if qs >= np.cos(np.deg2rad(self.precision)/2):
                reward += 9.

        elif self.steps_beyond_done is None:
            # episode just ended
            self.steps_beyond_done = 0
            if exceed:
                reward = -25.
            elif qs >= np.cos(np.deg2rad(self.precision)/2):
                reward = 50.
            else:
                reward = 0.

        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True."
                    " You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        self.qs_prev = qs

        return np.array(self.state0), reward, done, {}

    def reset(self, init_quat=None):

        self.nsteps = 0

        # Generate goal quaternion and set it as initial goal as well. this will be current-->goal orn.
        if init_quat is None:
            goal_aa = self._randomAxisAngle(self.min_angle_slew, self.max_angle_slew)
            if self.test:
                np.save("Savings/Manim/AxisAngle.npy", goal_aa)
            self.q_error_0 = self._axisAngleToQuat(goal_aa)
            self.q_error = self.q_error_0
        else:
            self.q_error_0 = init_quat
            if len(self.q_error_0) != 4:
                raise ValueError('invalid quat given')
            self.q_error = self.q_error_0

        # self.q_error_0_val = np.dot(self.alpha, np.square(self.q_error_0))

        # self.q_error_prev = self.q_error_0_val

        self.initial_angle = 2 * np.arccos(self.q_error[3])

        # Initial rate of change of error quat for state vector
        q_error_dot = np.array([0., 0., 0., 0.])

        # q ref is the reference quaternion, or quat from initial-->current orn.
        # self.q_ref = self.q_initial

        # Initial rate of change of ref quat for state vector
        # self.q_ref_dot = np.array([0., 0., 0., 0.])

        # start from rest (this is angular velocity)
        omega = np.array([0., 0., 0.])
        # start from de-tumbling conditions
        # omega = np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])

        self.state0 = (
            self.q_error[0], self.q_error[1], self.q_error[2], self.q_error[3],
            q_error_dot[0], q_error_dot[1], q_error_dot[2], q_error_dot[3],
            omega[0], omega[1], omega[2])

        self.steps_beyond_done = None

        qs = self.q_error[3]
        self.qs_prev = qs

        return np.array(self.state0)

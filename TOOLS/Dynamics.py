import numpy as np
from numpy.linalg import norm


# DEFINE THE DYNAMICS

def dynamics(t, state, I, M):
    """
    THE STATE EMBEDS THE ERROR QUATERNION FROM THE DESIRED ATTITUDE,
    ITS DERIVATIVE AND THE ANGULAR VELOCITY

    INPUT:
    - t: time vector
    - state: vector of the state variables(q, q_dot, om)
    - I: inertia matrix of the spacecraft
    - M: control torques applied, selected by the network

    OUPTUT:
    - dstate: derivative of the state vector

    AUTHOR: Lorenzo Capra - Politecnico di Milano
    """

    if len(state) != 11:
        raise Exception(f'The state vector must have 11 elements! Its length is {len(state)}')

    # Attitude error quaternion
    q = np.array([state[:4]])
    q /= norm(q)

    # Attitude error quaternion derivative
    # q_dot = state[4:8]

    # Angular velocities
    om = state[8:]

    OM = np.array([[0, om[2], -om[1], om[0]],
                   [-om[2], 0, om[0], om[1]],
                   [om[1], -om[0], 0, om[2]],
                   [-om[0], -om[1], -om[2], 0]])

    q_dot = 0.5 * np.dot(OM, np.transpose(q))
    q_dot = np.array([q_dot[0], q_dot[1], q_dot[2], q_dot[3]])
    q_dot = np.squeeze(q_dot)

    q_dot_dot = np.array([0, 0, 0, 0])

    dom_x = ((I[1] - I[2]) / I[0]) * om[1] * om[2] + M[0] / I[0]
    dom_y = ((I[2] - I[0]) / I[1]) * om[0] * om[2] + M[1] / I[1]
    dom_z = ((I[0] - I[1]) / I[2]) * om[1] * om[0] + M[2] / I[2]

    dom = np.array([dom_x, dom_y, dom_z])

    dstate = np.concatenate((q_dot, q_dot_dot, dom))

    if len(dstate) != 11:
        raise Exception(f'The state vector derivative must have 11 elements! Its length is {len(dstate)}')

    return dstate

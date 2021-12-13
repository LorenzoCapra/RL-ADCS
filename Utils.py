import gym
import torch
import numpy as np
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# USEFUL FUNCTIONS AND TOOLS:
def get_gae(
    rewards: list,
    values: list,
    is_terminals: list,
    gamma: float,
    lamda: float,
    ):
    """
    Takes: lists of rewards, state values, and 1-dones.
    Returns: list with generalized adversarial estimators.
    """
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = (rewards[i] + gamma * values[i + 1] * is_terminals[i] - values[i])
        gae = delta + gamma * lamda * is_terminals[i] * gae
        returns.insert(0, gae + values[i])

    return returns


def trajectories_data_generator(
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    batch_size,
    num_epochs,
    ):
    """data-generator."""
    data_len = states.size(0)
    for _ in range(num_epochs):
        for _ in range(data_len // batch_size):
            ids = np.random.choice(data_len, batch_size)
            yield states[ids, :], actions[ids], returns[ids], log_probs[ids], values[ids], advantages[ids]


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""

        action = torch.FloatTensor(action)
        action = 0.5*torch.tanh(action)
        action = action.detach().cpu().numpy().squeeze()

        return action


def quat2A(q_vect, qs):
    I = np.identity(3)
    q_skew = np.array([[0, -q_vect[2], q_vect[1]],
                       [q_vect[2], 0, -q_vect[0]],
                       [-q_vect[1], q_vect[0], 0]])

    # Transformation from quaternion to cosine matrix
    A = (qs ** 2 - np.dot(np.transpose(q_vect), q_vect)) * I + \
        2 * np.dot(q_vect, np.transpose(q_vect)) - 2 * qs * q_skew

    return A


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


def get_arrow(theta):
    x = np.cos(theta)
    y = np.sin(theta)
    z = 0
    u = np.sin(2*theta)
    v = np.sin(3*theta)
    w = np.cos(3*theta)
    return x, y, z, u, v, w


quiver = ax.quiver(*get_arrow(0))


def render_simulation(quaternion):
    q1, q2, q3, q4 = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    I = np.identity(3)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title('Attitude control simulation rendering')
    ax.set_box_aspect([1, 1, 1])

    zeros = [0.0, 0.0, 0.0]

    '''
    The frame is passed into the quiver method by rows, but they
    are being plotted by columns. So the 3 basis vectors of the frame
    are the columns of the 3x3 matrix
    '''

    # Inertial frame & Desired attitude
    ax.quiver(zeros, zeros, zeros, I[:, 0], I[:, 1], I[:, 2], color='black', label='Desired Attitude')
    # Labels
    ax.text(1.1, 0, 0, 'X', color='black')
    ax.text(0, 1.1, 0, 'Y', color='black')
    ax.text(0, 0, 1.1, 'Z', color='black')

    def update(i):
        global quiver
        quiver.remove()

        q_vect = np.array([q1[int(i)], q2[int(i)], q3[int(i)]])
        qs = q4[int(i)]

        # Transformation from quaternion to cosine matrix
        A = quat2A(q_vect, qs)
        # Orthonormalize the matrix
        A = A.dot(inv(sqrtm(A.T.dot(A))))

        # Plot of current attitude
        quiver = ax.quiver(zeros, zeros, zeros, A[:, 0], A[:, 1], A[:, 2], color='m', label='Current Attitude')
        ax.legend()

        return quiver,

    ax.view_init(elev=37, azim=45)

    ani = animation.FuncAnimation(fig, update, frames=500, interval=50)

    plt.show()

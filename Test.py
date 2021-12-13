from Environment import AttitudeControlEnv
from Utils import ActionNormalizer
from TOOLS.Agent import PPOAgent

# START THE TRAINING AND PLOT THE RESULTS:

if __name__ == '__main__':
    # DEFINE THE INITIAL CONDITIONS:
    q = [0, 0, 0, 0]
    q_dot = [0, 0, 0]
    om = [0, 0, 0]

    state0 = q + q_dot + om

    # ----------------------------------------------------------------------------------------------------------- #

    # DEFINE MAX SIMULATION LENGTH:
    max_simulation_length = 500  # number of actions taken: n/0.01 -> 1000

# ----------------------------------------------------------------------------------------------------------- #

    # DEFINE THE ENVIRONMENT:
    env = AttitudeControlEnv(state0, max_simulation_length, precision=0.25, test=True)
    env = ActionNormalizer(env)

# ----------------------------------------------------------------------------------------------------------- #

    # DEFINE THE AGENT:
    agent = PPOAgent(
        env,
        gamma=0.99,
        lamda=0.95,
        entropy_coef=0.002,
        epsilon=0.1,
        rollout_len=max_simulation_length,
        total_rollouts=500,
        update_timesteps=5000,
        num_epochs=8,
        batch_size=100,
        is_evaluate=False,
        solved_reward=None,
        actor_lr=1e-5,
        critic_lr=1e-4,
        regulate_entropy=(False, 100),
    )

# ----------------------------------------------------------------------------------------------------------- #

    # LOAD THE MODEL:
    agent.load_model(actor='Savings/Actors/actor4',
                     critic='Savings/Critics/critic4')

    # EVALUATE THE MODEL:
    print('---------------------------------------------------------------------------')
    agent.evaluate(n_tests=1, plot_q=False)
    print('---------------------------------------------------------------------------')

    # SAVE ARRAYS FOR MANIM ANIMATIONS:
    agent.SaveArrays()

    # RENDER THE SIMULATION:
    agent.Render()

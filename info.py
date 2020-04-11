
class Info:
    def __init__(self, env, brain_name, brain):
        self.env = env
        self.brain_name = brain_name
        self.brain = brain
        print("created Info")

    def print_info(self):
        # reset the environment
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))

        # number of actions
        action_size = self.brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space 
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

    def getInfo(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        action_size = self.brain.vector_action_space_size
        state = env_info.vector_observations[0]
        state_size = len(state)
        return action_size, state_size
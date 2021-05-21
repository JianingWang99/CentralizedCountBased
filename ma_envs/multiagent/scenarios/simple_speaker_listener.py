import numpy as np
from ma_envs.multiagent.core import World, Agent, Landmark, Flag
from ma_envs.multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08

        # add flags to show speaker's utterance
        world.flags = [Flag() for i in range(world.dim_c)]
        for i, flag in enumerate(world.flags):
            flag.name = 'flag %d' % i
            flag.collide = False
            flag.movable = False
            flag.size = 0.02

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)

        # set speaker at the left down position
        world.agents[0].state.p_pos = np.array([-0.75,-0.75])
        world.agents[0].state.p_vel = np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)
        world.agents[1].state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        world.agents[1].state.p_vel = np.zeros(world.dim_p)
        world.agents[1].state.c = np.zeros(world.dim_c)
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, flag in enumerate(world.flags):
            flag.color = np.array([0.95,0.95,0.95])
            flag.state.p_pos = np.array([(-0.65 + i*(0.05)), -0.65])
            flag.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        return -dist2
        # a = world.agents[0]
        # dist2 = np.sqrt(np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos)))
        # rew = 0 if dist2 < a.goal_b.size else -1
        # return rew/2

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color
        
        #TODO: do not show the color of the landmarks as the paper?
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)
        
        # speaker
        if not agent.movable:
            return np.concatenate(([goal_color], [0,0,0,0,0,0,0,0]), axis=None)
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
            
    def success_rate(self, agent, world):
        reach = 0
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sqrt(np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos)))
        if dist2 < a.goal_b.size+a.size:
            reach = 1
        return reach
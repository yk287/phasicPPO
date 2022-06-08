




import gym

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

from torch.distributions import Categorical

from collections import deque

import os.path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining


class PPO(nn.Module):
    def __init__(self, state_size, action_size, opts):
        super(PPO, self).__init__()

        self.opts = opts
        self.state_size = state_size
        self.action_size = action_size

        # list used to store data
        self.data = []

        self.layer = nn.Linear(self.state_size, self.opts.layer_1)
        self.policy = nn.Linear(self.opts.layer_1, self.action_size)
        self.value = nn.Linear(self.opts.layer_1, 1)

        # initialize the AC
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.opts.lr)

    def forward(self, x, softmax_dim = 0):
        '''
        Run through the policy model and get a probability of actions
        :param x: state
        :param softmax_dim:
        :return:
        '''

        if type(x) is np.ndarray:
            # convert to cuda
            x = torch.from_numpy(x).float().to(device)

        # feed the state through the model
        x = F.relu(self.layer(x))
        x = self.policy(x)

        prob = torch.exp(x - torch.max(x, dim=softmax_dim, keepdim=True)[0])
        prob = prob / torch.sum(prob, dim=softmax_dim, keepdim=True)

        # return prob that we'll sample actions from
        return prob

    def get_action(self, x):

        # forward
        prob = self.forward(x)
        # turn into categorical
        dist = Categorical(prob)

        # sample an action
        action = dist.sample()

        return action.item(), prob[action].item()

    def value_forward(self, x):
        '''
        Run through the value model to get the value of state x.
        :param x: state
        :return:
        '''

        x = F.relu(self.layer(x))
        v = self.value(x)
        return v

    def append_memory(self, transition):
        '''
        append trajectories into the memory
        :param transition:
        :return:
        '''
        self.data.append(transition)

    def reset_buffer(self):
        # reset the data list
        self.data = []

    def sample_memory(self):

        # create a list that will hold transitions
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        action_prob_list = []
        done_list = []

        for data in self.data:

            state, action, reward, next_state, action_prob, done = data

            state_list.append(state)
            action_list.append(action)
            reward_list.append([reward])
            next_state_list.append(next_state)
            action_prob_list.append(action_prob)
            done_mask = 0 if done else 1
            done_list.append([done_mask])

        # turn lists into tensors
        state = torch.tensor(state_list, dtype=torch.float).to(device)
        action = torch.tensor(action_list).to(device)
        reward = torch.tensor(reward_list).to(device)
        next_state = torch.tensor(next_state_list, dtype=torch.float).to(device)
        done_mask = torch.tensor(done_list, dtype=torch.float).to(device)
        action_prob = torch.tensor(action_prob_list).to(device)

        return state, action, reward, next_state, done_mask, action_prob

    def choose_mini_batch(self, mini_batch_size, states, actions, rewards, next_states, done_mask_, log_prob_, advantage_, returns):
        '''
        randomizes the batch and create a function that returns tuples when called.
        '''

        full_batch_size = len(states)

        # yield number of samples that are equal to full batch size // mini batch size
        for _ in range(full_batch_size // mini_batch_size):

            # randomly sample mini_batch_size size indices
            idx = np.random.randint(0, full_batch_size, mini_batch_size)

            yield states[idx], actions[idx], rewards[idx], next_states[idx], done_mask_[idx], \
                  log_prob_[idx], advantage_[idx], returns[idx]


    def GAE(self, state, action, reward, next_state, done, old_log_prob):
        '''
        Performs GAE calculations and then returns all the data needed for updating ppo model
        :return:
        '''

        # get the temporal difference target, reward + discount_rate * value of next_state
        td_target = reward + self.opts.gamma * self.value_forward(next_state) * done

        # pred vs target, eq 17 from GAE paper
        delta = td_target - self.value_forward(state)
        delta = delta.detach().cpu().numpy()

        advantage_list = []
        advantage = 0.0

        for idx in reversed(range(len(delta))):

            advantage = self.opts.gamma * self.opts.lambda_ * advantage * done[idx] + delta[idx][0]
            advantage_list.append([advantage])

        # reverse the order of the list
        advantage_list.reverse()
        # convert to pytorch tensor
        advantage_ = torch.tensor(advantage_list, dtype=torch.float).to(device)

        # advantage = sum of returns - value(state) in eq 18, so flip that around to get the monte carlo rewards
        # mc_returns get used as a target for updating value function
        mc_returns = advantage_ + self.value_forward(state)

        # normalize the advantage
        advantage_ = (advantage_ - advantage_.mean()) / (advantage_.std() + 1e-3)

        # return all the data needed to update the ppo model.
        return state, action, reward, next_state, done, old_log_prob, advantage_, mc_returns

    def train_ppo(self):

        # sample data from agent's trajectories
        state, action, reward, next_state, done, old_log_prob = self.sample_memory()

        # get the GAE values
        states, actions, rewards, next_states, dones, old_log_probs, advantages, mc_returns = self.GAE(state, action, reward, next_state, done, old_log_prob)

        # perform ppo_update
        self.ppo_update(states, actions, rewards, next_states, dones, old_log_probs, advantages, mc_returns)

    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, advantages, mc_returns):

        # update ppo model multiple times
        for i in range(self.opts.ppo_epochs):
            # get the mini batch samples
            for state,action,reward,next_state,done,old_action_prob,advantage,mc_return in self.choose_mini_batch(self.opts.batch, states, actions, rewards, next_states, dones, old_log_probs,advantages,mc_returns):

                # get the policy, which gives us the probability of actions being taken
                policy = self.forward(state, softmax_dim=1)
                # get the new probability of action
                p_action = policy.gather(1,action)
                # calculate the likelihood between old and new action probabilities
                ratio = torch.exp(torch.log(p_action) - torch.log(old_action_prob))  # a/b == exp(log(a)-log(b))
                # ppo objective with 2 surrogates
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.opts.epsilon, 1+self.opts.epsilon) * advantage

                # loss is the min between 2 surrogate objectives and also the difference between value(state) and monte carlo returns
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.value_forward(state), mc_return.detach())

                # take the gradient and update the model
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # reset the memory buffer
        self.reset_buffer()


def train(
        model,
        iters,
        last_100_scores,
        opts,
          ):

    # initialize the environments
    env = gym.make(opts.env_name)

    T_horizon = opts.batch * opts.horizon_multiplier  # set the batch size and how many batches to get per rollout

    # empty list to keep track of scores
    score_lst = []

    for eps in range(opts.epochs):

        # reset the score
        score = 0.0
        # reset the env
        state = env.reset()

        # keep iterating until we reach the number of samples that we want.
        for t in range(T_horizon):

            # get the probs
            action, a_prob = model.get_action(state)
            next_state, reward, done, info = env.step(action)

            # append the data
            model.append_memory((state, np.asarray([action]), reward / 100.0, next_state, np.asarray([a_prob]), done))

            # add the reward to score
            score += reward
            # if the env is done
            if done:
                # reset the environment
                state = (env.reset())
                # append the score
                score_lst.append(score)
                # append the score to deque
                last_100_scores.append(score)
                # reset the score
                score = 0
            else:
                # otherwise update state
                state = next_state

        # after we've collected T_horizon number of samples update the model
        model.train_ppo()

        # print every opts.print_every to keep track of how well the model is being trained
        if iters % opts.print_every == 0 and eps != 0:
            print("# of episode :{}, avg score : {:.2f}".format(iters, np.mean(last_100_scores)))
            score_lst = []

        # if the average score of the last 100 episode is great than or equal to opts.end_condition, then the env is solved
        if np.mean(last_100_scores) >= opts.win_condition:
            print("Environment Solved\n")
            print("# of episode :{}, avg score : {:.2f}".format(iters, np.mean(last_100_scores)))

        if eps == opts.train_iterations_per_step:
            break

    env.close()

    return last_100_scores,

def main(config, checkpoint_dir=None):

    step = 0
    opts = config['opts']

    running_score = deque(maxlen=100)

    # Init agent

    # initialize the environments
    env = gym.make(opts.env_name)

    # instantiate the model
    model = PPO(env.observation_space.shape[0], env.action_space.n, opts).to(device)

    if checkpoint_dir is not None:

        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        step = checkpoint["step"]

        if "lr" in config:
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = config["D_lr"]

        if "lambdas" in config:
            opts.lambdas = config['lambdas']

    running_score.append(0)

    while np.mean(running_score) < opts.target_is:

        running_score = train(
            model,
            step,
            running_score,
            opts,
            )

        step += 1
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                },
                path,
            )

        tune.report(iters=step, score=np.mean(running_score))

def pbt(opts):

    ray.init()

    # PBT scheduler
    scheduler = PopulationBasedTraining(
        perturbation_interval=opts.perturb_iter,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(1e-4, 1e-5),
        },
    )

    config = {
        "opts": opts,
        "use_gpu": True,
        "lr": tune.choice([0.00005, 0.00001, 0.000025]),
        "epsilon": tune.choice([0.1, .15, .20]),
        "ppo_epochs": tune.choice([3, 4, 5]),
        "horizon_multiplier": tune.choice([8, 16, 32]),
        "lambdas":tune.choice([0.5, 1.0, 1.5, 2.0])
        }

    reporter = CLIReporter(
        metric_columns=["iters", "scores"])

    analysis = tune.run(
        main,
        name="RL",
        scheduler=scheduler,
        resources_per_trial={"cpu": opts.cpu_use, "gpu": opts.gpu_use},
        verbose=1,
        stop={
            "training_iteration": opts.tune_iter,
        },
        metric="scores",
        mode="max",
        num_samples=opts.num_sample,
        progress_reporter=reporter,
        config=config
    )

    all_trials = analysis.trials
    checkpoint_paths = [
        os.path.join(analysis.get_best_checkpoint(t), "checkpoint")
        for t in all_trials
    ]

    best_trial = analysis.get_best_trial("scores", "max", "last-5-avg")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="scores")


if __name__ == "__main__":

    #load the options for testing out different configs
    from options import options
    options = options()
    opts = options.parse()

    pbt(opts)



















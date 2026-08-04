<!-- genesis_reward research_msg v1 (2026-08-04). Frozen domain brief, shared
     identically by every experiment arm (e0 / e3r / e6p) so that it is not
     itself an ablation variable. Contains terrain and published facts only:
     no reward terms, no coefficients, no recipe. -->

# Domain brief: reward design for legged locomotion

## Why this task exists

Reward design is the standard example of a task where the objective is easy to
state and hard to optimise. The fitness here is a single exponential in the
forward-velocity error, and the seed genome rewards precisely that. If direct
optimisation of the objective were the right answer, the seed would already be
optimal and there would be nothing to search. Published results on this exact
HyperAgents environment reach roughly 0.79 average fitness, which is well above
what the naive seed obtains — so the gap between "reward the objective" and
"reward what makes the objective learnable" is real and large.

## The mechanism behind that gap

Three properties of the training loop, all fixed and all outside the genome:

1. **Exploration is driven by the reward's gradient, not its value.** A reward
   that is flat over most of the state space gives PPO nothing to climb. Early
   in training the robot cannot walk at all, so any term that only pays out
   near the target velocity pays out almost never.
2. **Early termination truncates the return.** Roll or pitch beyond 10 degrees
   ends the episode. Behaviour that scores well per step but ends the episode
   early collects less total fitness than behaviour that scores modestly and
   survives 20 seconds.
3. **The policy has 101 iterations and no more.** Every candidate trains under
   an identical budget, so a reward that would eventually converge to a better
   gait but has not converged by iteration 101 measures as worse. Learning
   speed is part of what is being selected for, whether or not that was intended.

## What varies between evaluations

The same genome trained twice does not score identically: PPO is stochastic,
the command sequence is resampled per episode, and the physics run in
single precision on the GPU. The reported standard error over completed
episodes is the only variance estimate available; treat two candidates whose
intervals overlap heavily as unranked rather than ordered.

## What is already known not to work

- Rescaling the seed's single term. Multiplying the whole reward by a constant
  leaves the policy gradient direction unchanged up to the learning-rate
  schedule, which PPO adapts anyway (`desired_kl = 0.01`, adaptive schedule).
- Rewarding quantities the robot cannot observe. The policy's 45-dimensional
  observation is angular velocity, projected gravity, commands, joint positions
  and velocities, and last actions. A reward defined on something outside that
  set can still be optimised — the critic sees the same observation — but the
  actor has no way to condition on it.

## Deliberately not in this brief

Which additional terms help, what their relative weights should be, what
temperature an exponential kernel wants, and whether penalties should be
absolute or squared. Those are the search's job; supplying them here would
make every arm score the same and measure nothing about the framework.

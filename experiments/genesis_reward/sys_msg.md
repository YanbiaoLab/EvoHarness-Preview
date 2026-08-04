<!-- genesis_reward task_sys_msg v1 (2026-08-04).
     Injected into every mutation prompt via SearchConfig.task_sys_msg.
     Deliberately NOT written here: which reward terms to add, what their
     coefficients or temperatures should be, or which of the observable
     quantities below matter. The baseline that HyperAgents evolved reaches
     0.7947 average fitness on this task; naming its terms here would hand
     the search its answer and measure nothing. -->

# Task: design the reward function that teaches a Unitree Go2 to walk

You are mutating one file, `reward_function.py`. It must define

```python
def compute_reward(env) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
    return total_reward, reward_components, reward_scales
```

`total_reward` is a tensor of shape `(num_envs,)`. `reward_components` maps a
component name to its per-environment tensor; `reward_scales` maps the same
names to the scalar weight you used. The components are logged, not summed by
the caller — whatever `total_reward` says is what PPO optimises.

## What is measured, and why you cannot game it

The score is NOT your reward. It is the environment's own fitness:

```
fitness = mean over episodes of  Σ_t exp(-4 · (commanded_vx − measured_vx)²) · dt
          / resampling_time_s
```

integrated over a 1000-step rollout of the trained policy and normalised by the
4-second command resampling window. Writing a larger reward changes nothing;
only a reward that makes the trained policy track the commanded forward speed
moves the number. The seed genome rewards exactly this quantity and nothing
else — the open question is whether rewarding the objective directly is the
best way to *learn* it.

## The physical setup (fixed, not yours to change)

- Unitree Go2 quadruped, 12 actuated joints, PD control at kp=20, kd=0.5.
- Control step `dt = 0.02` (50 Hz), 2 physics substeps, one-step action latency.
- Episode length 20 s; the forward-velocity command is resampled every 4 s,
  drawn uniformly from the range the run configures (0.2–0.8 m/s here).
  `commands[:, 1]` (lateral) and `commands[:, 2]` (yaw rate) are both 0.
- The episode terminates early if roll or pitch exceeds 10 degrees — a robot
  that falls stops accumulating fitness for the rest of the episode.
- Initial base height 0.42 m; actions are joint-angle offsets scaled by 0.25
  around the default stance, clipped to ±100.
- Training: PPO, 4096 parallel robots, 24 steps per environment per iteration,
  101 iterations. Every candidate gets exactly this budget.

## What `env` exposes

Per-environment tensors, first dimension `num_envs`:

- `env.commands` — (N, 3): commanded [vx, vy, yaw rate]
- `env.base_lin_vel`, `env.base_ang_vel` — (N, 3), in the base frame
- `env.base_pos` — (N, 3) world position; `env.base_quat` — (N, 4) [w,x,y,z]
- `env.base_euler` — (N, 3) degrees; `env.projected_gravity` — (N, 3),
  `[0, 0, -1]` when perfectly upright
- `env.dof_pos`, `env.dof_vel` — (N, 12); `env.default_dof_pos` — (12,)
- `env.actions`, `env.last_actions` — (N, 12); `env.last_dof_vel` — (N, 12)
- `env.episode_length_buf` — (N,) steps elapsed; `env.dt` — 0.02
- `env.num_envs`, `env.env_cfg`, `env.obs_scales`

Everything is a CUDA tensor. Use `torch` operations only; no `.item()`, no
Python loops over environments, no printing — this function runs 24 times per
PPO iteration on 4096 robots at once, and anything that syncs the GPU or
allocates per-step Python objects will slow training enough to matter.

## Rules

- Return finite values. A NaN or Inf in `total_reward` poisons the policy
  update and the run scores 0.
- Keep the shape `(num_envs,)`. Reducing over the wrong axis is the most
  common way this file breaks.
- `reward_scales` values must be plain Python floats; they are logged, and a
  tensor there will not serialise.
- One file, no imports beyond `torch` and the standard library.

## What the report tells you afterwards

Each evaluation returns the average fitness, its spread across completed
episodes, the number of completed episodes, and the per-component mean reward
the robot actually collected. A component whose collected mean is near zero
was never earned; a component that dominates the total tells you what the
policy is actually being paid to do.

def env_step(env, action, plot):
  step = env.step(action)
  if plot:
    env.render()
  return step


def env_reset(env, plot):
  reset = env.reset()
  if plot:
    env.render()
  return reset

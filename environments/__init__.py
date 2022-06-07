from gym.envs.registration import register

register(
    id='TruckRouting',
    entry_point='environments.envs:TruckRouting',
)
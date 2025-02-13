import io
import cProfile
import tracemalloc
import gymnasium as gym
from gymnasium.spaces import Text

GB = 1024 * 1024 * 1024
DEFAULT_CONFIG = {
    "max_input_len": 32000,    # longest length of the code
    "max_time_cost": 60,       # longest time of execution
    "max_memory_cost": 4 * GB, # largest available memory
    "entry_point": "env_main"
}

class SimpleEvaluator(gym.Env):
    def __init__(self, config=DEFAULT_CONFIG):
        self.action_space = Text(config["max_input_len"])
        self.entry_point = config["entry_point"]
        self.max_time_cost = config["max_time_cost"]
        self.max_memory_cost = config["max_memory_cost"]
        self.observation_space = Text(2048)
        self.reward = 0

    def reset(self, *, seed=None, options=None):
        self.reward = 0
        return "", {"env_state": "reset"}

    def step(self, action):
        assert self.entry_point in action
        exec(action)
        # Start tracing memory allocations
        tracemalloc.start()
        tmp = eval(f"{self.entry_point}()")
        # Take a snapshot
        snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.get_traced_memory()
        # Stop tracing memory allocations
        tracemalloc.stop()
        # memory_reward = peak - current
        memory_reward = peak
        # Analyze the snapshot to see memory usage
        top_stats = snapshot.statistics('lineno')
        memory_usage = "[ Detailed traceback for the top memory consumer ]\n"
        for stat in top_stats[:1]:
            memory_usage += '\n'.join(stat.traceback.format()) + '\n'
        pr = cProfile.Profile()
        pr.enable()
        tmp = eval(f"{self.entry_point}()")
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        time_usage = s.getvalue()
        time_reward = float(
            time_usage.split('function calls in ')[1].split(' ')[0])
        observation = (
            f"# Space complexity\n\n{memory_usage}\n\n---\n"
            f"# Time complexity\n\n{time_usage}")
        normalized_time_cost = min(time_reward / self.max_time_cost, 1)
        normalized_memory_cost = min(time_reward / self.max_memory_cost, 1)
        # total_cost = (time_weight * normalized_time_cost) + (memory_weight * normalized_memory_cost)
        # reward = -total_cost
        terminated = False
        truncated = False
        infos = {}
        reward = -(normalized_time_cost + normalized_memory_cost)
        return (
            observation,
            reward,
            terminated,
            truncated,
            infos,
        )
import io
import sys
import pstats
import cProfile
import traceback
import tracemalloc
import gymnasium as gym
from gymnasium.spaces import Text

GB = 1024 * 1024 * 1024
DEFAULT_CONFIG = {
    "max_input_len": 32000,    # longest length of the code
    "max_time_cost": 60,       # longest time of execution
    "max_memory_cost": 4 * GB, # largest available memory
    "entry_point": "env_main",
    "exception_reward": -9
}

class SimpleEvaluator(gym.Env):
    """
    A Gymnasium environment for evaluating the performance (time and memory) of Python code.

    This environment allows an AI agent to submit Python code (as a string) and receive
    feedback on its execution time and memory usage. It uses `cProfile` for time profiling
    and `tracemalloc` for memory profiling. The environment provides a reward signal
    based on the time and memory costs, encouraging the agent to optimize its code.

    Attributes:
        action_space (gym.spaces.Text): The action space, which accepts Python code as a string.
                                         The maximum length of the code is defined by `max_input_len`.
        observation_space (gym.spaces.Text): The observation space, which provides profiling results
                                              (time and memory usage) as a string.
        entry_point (str): The name of the function to be executed within the submitted code.
                           This function *must* be defined within the submitted code.  Defaults to "env_main".
        max_time_cost (float): The maximum allowed execution time (in seconds).  Used for reward normalization.
        max_memory_cost (int): The maximum allowed memory usage (in bytes). Used for reward normalization.
        time_weight (float):  Weight for the time cost in the reward calculation.
        memory_weight (float): Weight for the memory cost in the reward calculation.
        reward (float):  The current reward value.

    Configuration (config dictionary):
        max_input_len (int): Maximum length of the input code string (default: 32000).
        max_time_cost (float): Maximum allowed execution time in seconds (default: 60).
        max_memory_cost (int): Maximum allowed memory usage in bytes (default: 4 * GB).
        entry_point (str): The name of the function to execute (default: "env_main").
        time_weight (float): Weight for time cost in reward calculation (default: 1).
        memory_weight (float): Weight for memory cost in reward calculation (default: 1).
        exception_reward (float): Negative number reward for exception in code (default: -9).
                                  The number will multiply with `max(time_weight, memory_weight)`.

    Methods:
        reset(seed=None, options=None): Resets the environment to its initial state.
        step(action): Executes the given Python code, profiles its performance, and returns the results.

    Example Usage:

    ```python
    from python_perf_env import SimpleEvaluator

    # Create the environment
    env = SimpleEvaluator()  # Uses the DEFAULT_CONFIG

    # Define the code to be executed (must include the entry_point function)
    code = '''
    import time

    def env_main():
        i = 0
        for _ in range(1000000):
            i += i  # Simulate some work
            time.sleep(0.01)  # Simulate a delay
        return i
    '''

    # Take a step
    observation, reward, terminated, truncated, info = env.step(code)

    print("Observation:")
    print(observation)
    print("Reward:", reward)
    assert terminated == False
    assert truncated == False
    assert not info # Dict is Empty

    # Example using a custom configuration:
    GB = 1024 * 1024 * 1024
    custom_config = {
        "max_input_len": 2048,
        "max_time_cost": 1,  # The maximum allowed execution time (in seconds)
        "max_memory_cost": 1 * GB,  # The maximum allowed memory usage (in bytes)
        "entry_point": "my_function",
        "exception_reward": -9
    }

    env_custom = SimpleEvaluator(config=custom_config)
    observation, info = env_custom.reset()

    code_custom = '''
    import time

    def my_function():
        time.sleep(0.5)  # Simulate a delay
        m = [[0]*1000]*1000 # Create a matrix
        return m
    '''
    observation, reward, terminated, truncated, info = env_custom.step(code_custom)
    print("\nCustom Config Example:")
    print("Observation:", observation)
    print("Reward:", reward)
    ```
    """
    def __init__(self, config=DEFAULT_CONFIG):
        """
        Initializes the SimpleEvaluator environment.

        Args:
            config (dict, optional): A dictionary containing configuration parameters.
                                     Defaults to DEFAULT_CONFIG.
        """
        self.action_space = Text(config["max_input_len"])
        self.entry_point = config.get("entry_point")
        if self.entry_point is None:
            self.entry_point = DEFAULT_CONFIG["entry_point"]
        self.max_time_cost = config["max_time_cost"]
        self.max_memory_cost = int(config["max_memory_cost"])
        self.time_weight = 1
        self.memory_weight = 1
        if "time_weight" in config and "memory_weight" in config:
            self.time_weight = config["time_weight"]
            self.memory_weight = config["memory_weight"]
        self.exception_reward = (
            config["exception_reward"] *
            max(self.time_weight, self.memory_weight)
        )
        assert self.exception_reward < 0, "Exception reward must be negative"
        assert min(self.time_weight, self.memory_weight) > 0, "Time and memory weights must be positive"
        assert self.max_memory_cost > 0, "Max memory cost must be positive"
        assert self.max_time_cost > 0, "Max time cost must be positive"
        assert len(self.entry_point) > 0, "Entry point must be a non-empty string"
        assert config["max_input_len"] > 8 + len(self.entry_point), "Max input length must be greater than the length of the name of function"
        self.observation_space = Text(2048)
        self.reward = 0

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment.

        Returns:
            tuple: A tuple containing the initial observation (empty string) and an info dictionary
                   with the key "env_state" set to "reset".
        """
        self.reward = 0
        return "", {"env_state": "reset"}

    def step(self, action):
        """
        Executes the provided Python code, profiles its time and memory usage, and returns the profiling results.

        Args:
            action (str): A string containing the Python code to be executed.  The code *must*
                          define a function with the name specified in `self.entry_point`.

        Returns:
            tuple: A tuple containing:
                - observation (str): A string containing the profiling results (time and memory usage).
                - reward (float): The calculated reward based on time and memory costs.
                - terminated (bool): Always False, as the environment does not have a terminal state.
                - truncated (bool): Always False, as the environment does not have a truncation condition.
                - infos (dict): Always an empty dictionary, as the environment does not have extra output.
        """
        assert self.entry_point in action, f"Entry point '{self.entry_point}' not found in the provided code."
        try:
            exec(action, locals())
            # Start tracing memory allocations
            tracemalloc.start()
            executed_result = eval(f"{self.entry_point}()")
            _, memory_reward = tracemalloc.get_traced_memory()
            # Stop tracing memory allocations
            tracemalloc.stop()
            # Start tracing time costs
            cProfile.runctx(f"{self.entry_point}()", None, locals(), "eval_stats")
            s = io.StringIO()
            ps = pstats.Stats("eval_stats", stream=s).sort_stats('tottime').print_stats()
            time_usage = s.getvalue()
        except Exception as e:
            observation = traceback.format_exc()
            reward = self.exception_reward
            return (
                observation,
                reward,
                False,
                False,
                {}
            )
        # Analyze the snapshot to see memory usage
        memory_usage = f"Peak memory usage: {memory_reward} bytes ({memory_reward/1024/1024} MiB)"
        try:
            time_reward = float(
                time_usage.split('function calls in ')[1].split(' ')[0])
        except Exception as e:
            time_reward = self.max_time_cost # Assign max cost if parsing fails
        observation = (
            f"# Output\n{executed_result}\n\n---\n\n"
            f"# Space complexity\n\n{memory_usage}\n\n---\n\n"
            f"# Time complexity\n\n{time_usage}")
        normalized_time_cost = min(time_reward / self.max_time_cost, 1)
        normalized_memory_cost = min(memory_reward / self.max_memory_cost, 1)
        total_cost = (
            self.time_weight * normalized_time_cost
        ) + (
            self.memory_weight * normalized_memory_cost
        )
        reward = -total_cost
        terminated = False
        truncated = False
        infos = {}
        return (
            observation,
            reward,
            terminated,
            truncated,
            infos,
        )


class TestDrivenEvaluator(SimpleEvaluator):
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__(config)
        self.unittest_code = config.get("unittest")
        assert self.unittest_code, "unittest_code must be included in config"
        assert len(self.unittest_code) > 20, "unittest_code must be a non-empty string"
        self.test_execenv = {}

    def reset(self, *, seed=None, options=None):
        obs_tuple = super().reset(seed=seed, options=options)
        self.test_execenv = {}
        return obs_tuple

    def step(self, action):
        observation, reward, terminated, truncated, infos = super().step(action)
        try:
            exec(action, self.test_execenv)
            # Capture stdout for unittest results
            output = io.StringIO()
            sys.stdout = output
            # Execute the unit tests
            exec(self.unittest_code, self.test_execenv)
            exec("unittest.main(exit=False)", self.test_execenv)
            sys.stdout = sys.__stdout__
            unittest_output = output.getvalue()
        except Exception as e:
            unittest_output = f"Unit test execution failed:\n{traceback.format_exc()}"
        obs = observation.split("# Space complexity")
        obs[0] = f"# Unit Test Output\n{unittest_output}\n\n---\n\n"
        observation = "# Space complexity".join(obs)
        return (
            observation,
            reward,
            terminated,
            truncated,
            infos,
        )


class SecureEvaluator(gym.Env):
    """
    A Gymnasium environment for evaluating the performance (time and memory) of Python code.
    The Python code will be executed in a container environment instead of host machine 
    which makes it more secure as no file system operation will affect the host machine.
    """
    def __init__(self, config=DEFAULT_CONFIG):
        raise NotImplemented

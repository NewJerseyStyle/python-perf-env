# python-perf-env

[![PyPI version](https://badge.fury.io/py/python-perf-env.svg)](https://badge.fury.io/py/python-perf-env)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Python package providing an environment for AI agents to test their code and receive detailed profiling feedback on execution time and memory usage.

## Installation

Install the package using pip:

```bash
pip install python-perf-env
```

## Quick Start / Get Started

This guide provides a quick introduction to using the `python_perf_env` package. It demonstrates how to create and interact with the `SimpleEvaluator` environment.

### 1. Import the necessary modules:

```python
from python_perf_env import SimpleEvaluator
```

### 2. Create the environment:

You can create the environment using the default configuration:

```python
env = SimpleEvaluator()
```

Or, you can customize the environment by providing a configuration dictionary:

```python
# Say we want focused on optimizing algorithm on time
custom_config = {
    "max_time_cost": 1,  # The maximum allowed execution time (in seconds)
    "max_memory_cost": 1 * GB,  # The maximum allowed memory usage (in bytes)
    "time_weight": 2, # Weight for time cost in reward calculation (default: 1).
    "memory_weight": 0.5 # Weight for memory cost in reward calculation (default: 1).
    "entry_point": "my_function",
}
env_custom = SimpleEvaluator(config=custom_config)
```

### 3. Reset the environment:

Reset the environment to get the initial observation:

```python
env.reset()
```

### 4. Define the code to be executed:

The code you submit to the environment *must* include a function with the name specified in the `entry_point` configuration (defaults to `"env_main"`).

```python
code = """
import time

def env_main():
    start_time = time.time()
    for _ in range(1000000):
        pass  # Simulate some work
    end_time = time.time()
    return end_time - start_time  # Return the execution time
"""
```

### 5. Take a step in the environment:

Submit the Python code as an action to the `step()` method:

```python
observation, reward, terminated, truncated, info = env.step(code)

print("Observation:")
print(observation)
print("Reward:", reward)
```

## Further Information

*   **Documentation:**  Refer to the docstrings within the code for detailed information about the `SimpleEvaluator` class, its attributes, and its methods.
*   **Security:**  *Always* prioritize security when using this environment as it does not provide encapsulated environment for code execution. Consider providing secure environment using container in future versions.
*   **Profiling Output:** The format of the profiling output might need to be parsed by your AI agent.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
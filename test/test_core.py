import unittest

from python_perf_env import SimpleEvaluator

GB = 1024 * 1024 * 1024

class Testing(unittest.TestCase):
    def test_normal(self):
        env = SimpleEvaluator(config={
            "max_input_len": 2048,
            "max_time_cost": 1,
            "max_memory_cost": 1 * GB,
            "exception_reward": -9
        })
        
        code = """import time

def env_main():
    for _ in range(1000000):
        pass  # Simulate some work"""
        obs, reward, _, _, _ = env.step(code)
        self.assertFalse('error' in obs.lower())

    def test_exception(self):
        time_weight = 2
        exception_reward = -9
        env = SimpleEvaluator(config={
            "max_input_len": 2048,
            "max_time_cost": 1,
            "max_memory_cost": 1 * GB,
            "time_weight": 2,
            "memory_weight": 0.5,
            "exception_reward": exception_reward
        })
        
        code = """import time

def env_main():
    for _ in range(1000000):
        pass  # Simulate some work
    return end_time - start_time  # Return the execution time"""
        obs, reward, _, _, _ = env.step(code)
        self.assertTrue('error' in obs.lower())
        self.assertEqual(reward, exception_reward * time_weight)

    def test_timeout(self):
        env = SimpleEvaluator(config={
            "max_input_len": 2048,
            "max_time_cost": 1,
            "max_memory_cost": 1 * GB,
            "time_weight": 2,
            "memory_weight": 0.5,
            "exception_reward": -1
        })
        
        code = """import time

def env_main():
    for _ in range(2):
        time.sleep(1.1)"""
        obs, reward, _, _, _ = env.step(code)
        self.assertFalse('error' in obs.lower())
        self.assertLess(reward, -2)
        self.assertGreater(reward, -3)

    def test_outofmemory(self):
        env = SimpleEvaluator(config={
            "max_input_len": 2048,
            "max_time_cost": 1,
            "max_memory_cost": 10, # 10 bytes upperlimit
            "time_weight": 1,
            "memory_weight": 4,
            "exception_reward": -1
        })
        
        code = """import time

def env_main():
    for _ in range(7):
        time.sleep(0.1)
    return [1]*100"""
        obs, reward, _, _, _ = env.step(code)
        self.assertFalse('error' in obs.lower())
        self.assertLess(reward, -4)
        self.assertGreater(reward, -5)


if __name__ == '__main__':
    unittest.main()
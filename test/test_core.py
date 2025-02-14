import unittest

from python_perf_env import SimpleEvaluator

GB = 1024 * 1024 * 1024

class Testing(unittest.TestCase):
    def test_normal(self):
        env = SimpleEvaluator(config={
            "max_time_cost": 1,
            "max_memory_cost": 1 * GB,
        })
        
        code = """import time

def env_main():
    for _ in range(1000000):
        pass  # Simulate some work"""
        obs, reward, _, _, _ = env.step(code)
        self.assertFalse('error' in obs)

    def test_exception(self):
        time_weight = 2
        env = SimpleEvaluator(config={
            "max_time_cost": 1,
            "max_memory_cost": 1 * GB,
            "time_weight": time_weight,
            "memory_weight": 0.5,
        })
        
        code = """import time

def env_main():
    for _ in range(1000000):
        pass  # Simulate some work
    return end_time - start_time  # Return the execution time"""
        obs, reward, _, _, _ = env.step(code)
        self.assertTrue('error' in obs)
        self.assertGreater(reward, -9 * time_weight)

    def test_timeout(self):
        env = SimpleEvaluator(config={
            "max_time_cost": 1,
            "max_memory_cost": 1 * GB,
            "time_weight": 2,
            "memory_weight": 0.5,
        })
        
        code = """import time

def env_main():
    for _ in range(1000000):
        time.sleep(0.1)"""
        obs, reward, _, _, _ = env.step(code)
        self.assertFalse('error' in obs)
        self.assertGreater(reward, -2)
        self.assertLess(reward, -3)

    def test_outofmemory(self):
        env = SimpleEvaluator(config={
            "max_time_cost": 1,
            "max_memory_cost": 10, # 10 bytes upperlimit
            "time_weight": 1,
            "memory_weight": 4,
        })
        
        code = """import time

def env_main():
    for _ in range(1000000):
        time.sleep(0.1)"""
        obs, reward, _, _, _ = env.step(code)
        self.assertFalse('error' in obs)
        self.assertGreater(reward, -4)
        self.assertLess(reward, -5)


if __name__ == '__main__':
    unittest.main()
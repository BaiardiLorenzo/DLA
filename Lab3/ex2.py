from lab3.cli import reinforce_main

if __name__ == "__main__":
    reinforce_main("ex2", baselines=("none", "standardized", "value"))

import gym

evo = 'ALE/Breakout-v5'

env = gym.make(evo, render_mode='human')
env.reset()

for _ in range(1000):
    #env.render(mode='human')
    env.step(env.action_space.sample())

env.close()

"""
MAX_PRIME = 100

def getty():
    sieve = [True] * MAX_PRIME
    for i in range(2, MAX_PRIME):
        if sieve[i]:
            print(i)
            for j in range(i * i, MAX_PRIME, i):
                sieve[j] = False


if __name__ == "__main__":
    print("Hello World")
    
    getty()
"""

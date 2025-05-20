import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")  # Visual rendering
observation, info = env.reset(seed=42)
episode_count = 0
step_count = 0
total_steps = []  # Track steps per episode

for _ in range(1000):  # Run for 1000 steps total
    env.render()
    
    # Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    pole_angle, pole_angular_velocity = observation[2], observation[3]
    
    # Fixed strategy: Combine pole angle and angular velocity
    k = 0.5  # Weight for angular velocity
    decision = pole_angle + k * pole_angular_velocity
    
    # Action: Push left (0) if decision < 0, right (1) if decision > 0
    action = 1 if decision > 0 else 0
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    
    # Print observation for debugging
    print(f'Episode {episode_count}, Step {step_count}: observation={observation}')
    
    # Check if episode is done
    if terminated or truncated:
        print(f'Episode {episode_count} done after {step_count} steps')
        total_steps.append(step_count)
        observation, info = env.reset()
        episode_count += 1
        step_count = 0

# Print average steps per episode
if total_steps:
    print(f'Average steps per episode: {sum(total_steps) / len(total_steps):.2f}')
env.close()

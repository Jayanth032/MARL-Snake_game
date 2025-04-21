import pygame
import sys
from game.snake_env import SnakeEnv

def run_timed_competition(env, duration=120):
    """Run a timed competition between the snakes"""
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    remaining = duration
    
    # Reset environment and scores
    state = env.reset()
    env.scores = [0, 0]  # Explicitly reset scores
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        elapsed = (current_time - start_time) // 1000
        remaining = max(0, duration - elapsed)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Get greedy actions (no exploration)
        action1 = env.choose_action(state, env.q_table1, epsilon=0)
        action2 = env.choose_action(state, env.q_table2, epsilon=0)
        
        # Step environment
        state, _, done, _ = env.step(action1, action2)
        
        # Render with timer and scores
        env.render(show_timer=True, elapsed_time=elapsed)
        clock.tick(env.fps)
        
        if remaining <= 0 or done:
            running = False
    
    # Determine winner
    if env.scores[0] > env.scores[1]:
        result = "GREEN SNAKE WINS!"
    elif env.scores[1] > env.scores[0]:
        result = "BLUE SNAKE WINS!"
    else:
        result = "IT'S A TIE!"
    
    # Display results
    result_text = env.font.render(result, True, (255, 255, 0))
    env.screen.blit(result_text, (env.screen_width//2 - 100, env.screen_height//2))
    pygame.display.flip()
    pygame.time.wait(3000)  # Show results for 3 seconds

def main():
    # Initialize pygame
    pygame.init()
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    FPS = 10

    # Create environment
    env = SnakeEnv(SCREEN_WIDTH, SCREEN_HEIGHT, FPS)
    clock = pygame.time.Clock()

    # Training parameters
    NUM_EPISODES = 2000
    RENDER_EVERY = 50  # Render every N episodes
    EPSILON_START = 0.5
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995

    # Track best performance
    best_score = -float('inf')
    best_tables = None

    # Training loop
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        # Calculate current epsilon
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
        
        while not done:
            # Get actions
            action1 = env.choose_action(state, env.q_table1, epsilon)
            action2 = env.choose_action(state, env.q_table2, epsilon)
            
            # Step environment
            next_state, (reward1, reward2), done, _ = env.step(action1, action2)
            total_reward += (reward1 + reward2)
            
            # Update Q-tables
            env.update_q_table(state, action1, reward1, next_state, env.q_table1)
            env.update_q_table(state, action2, reward2, next_state, env.q_table2)
            
            state = next_state
            
            # Render occasionally
            if episode % RENDER_EVERY == 0:
                env.render()
                clock.tick(FPS)
                
            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.save_training()
                    pygame.quit()
                    sys.exit()
        
        # Track best performance
        if total_reward > best_score:
            best_score = total_reward
            best_tables = (env.q_table1.copy(), env.q_table2.copy())
        
        # Print episode stats
        if episode % 10 == 0:
            print(f"Episode {episode}: Total R {total_reward:.1f}, ε {epsilon:.3f}")
            
        # Periodic saving
        if episode % 100 == 0 and best_tables:
            env.q_table1, env.q_table2 = best_tables
            env.save_training(f"snake_checkpoint_ep{episode}.pkl")

    # Save final training
    env.save_training("snake_final_training.pkl")
    print("Training complete!")

    # Run competition
    run_timed_competition(env, duration=120)

    # Option to run multiple competitions
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press R to replay
                    run_timed_competition(env, duration=120)
                if event.key == pygame.K_q:  # Press Q to quit
                    pygame.quit()
                    sys.exit()

if __name__ == "__main__":
    main()
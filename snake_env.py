import pygame
import random
import numpy as np
import pickle
from datetime import datetime
from game.snake import Snake
from game.food import Food

class SnakeEnv:
    def __init__(self, screen_width, screen_height, fps):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps
        
        # Initialize game elements
        self.snake1 = Snake((100, 100), (0, 255, 0), screen_width, screen_height)
        self.snake2 = Snake((300, 300), (0, 0, 255), screen_width, screen_height)
        self.food = Food(screen_width, screen_height, num_foods=8)  # Multiple foods
        
        # Q-learning parameters
        self.q_table1 = {}
        self.q_table2 = {}
        self.alpha = 0.2  # Learning rate
        self.gamma = 0.95  # Discount factor
        
        # Enhanced reward system
        self.food_reward = 10
        self.collide_reward = -15
        self.step_reward = -0.05
        self.approach_reward = 0.5
        self.retreat_penalty = -0.3
        self.opponent_block_reward = 2
        
        # Competition tracking
        self.scores = [0, 0]
        self.game_time = 120  # 2 minutes in seconds
        self.last_food_time = pygame.time.get_ticks()
        self.food_spawn_interval = 5000  # 5 seconds in milliseconds
        self.max_foods = 15
        
        # Action mapping
        self.ACTION_MAP = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
        
        # Display setup
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.font = pygame.font.SysFont(None, 36)
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.snake1 = Snake((100, 100), (0, 255, 0), self.screen_width, self.screen_height)
        self.snake2 = Snake((300, 300), (0, 0, 255), self.screen_width, self.screen_height)
        self.food = Food(self.screen_width, self.screen_height, num_foods=8)
        self.scores = [0, 0]
        return self.get_state()
    
    def get_state(self):
        """Enhanced state representation with relative food positions and dangers"""
        head1 = self.snake1.body[0]
        head2 = self.snake2.body[0]
        
        # Find nearest food for each snake
        nearest_food1 = min(self.food.positions, 
                           key=lambda f: abs(f[0]-head1[0]) + abs(f[1]-head1[1]))
        nearest_food2 = min(self.food.positions, 
                           key=lambda f: abs(f[0]-head2[0]) + abs(f[1]-head2[1]))
        
        return (
            head1, self.snake1.direction,
            (1 if nearest_food1[0] > head1[0] else (-1 if nearest_food1[0] < head1[0] else 0)),
            (1 if nearest_food1[1] > head1[1] else (-1 if nearest_food1[1] < head1[1] else 0)),
            self._get_dangers(self.snake1),
            head2, self.snake2.direction,
            (1 if nearest_food2[0] > head2[0] else (-1 if nearest_food2[0] < head2[0] else 0)),
            (1 if nearest_food2[1] > head2[1] else (-1 if nearest_food2[1] < head2[1] else 0)),
            self._get_dangers(self.snake2)
        )
    
    def _get_dangers(self, snake):
        """Check for dangers in all 4 directions"""
        head = snake.body[0]
        dangers = [0, 0, 0, 0]  # Up, Down, Left, Right
        
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            new_pos = (
                (head[0] + dx * snake.block_size) % self.screen_width,
                (head[1] + dy * snake.block_size) % self.screen_height
            )
            
            # Check self-collision
            if new_pos in snake.body[1:]:
                dangers[i] = 1
                
            # Check other snake collision
            other_snake = self.snake2 if snake == self.snake1 else self.snake1
            if new_pos in other_snake.body:
                dangers[i] = 1
                
        return tuple(dangers)
    
    def step(self, action1, action2):
        """Execute one time step in the environment"""
        current_time = pygame.time.get_ticks()
        
        # Spawn new food periodically (changed from add_food to generate_foods)
        if current_time - self.last_food_time > self.food_spawn_interval:
            if len(self.food.positions) < self.max_foods:
                self.food.generate_foods()  # Changed this line
            self.last_food_time = current_time
        
        # ... rest of your step method remains the same ...
        # Store previous distances to food
        prev_dist1 = min(self._distance(self.snake1.body[0], food) for food in self.food.positions)
        prev_dist2 = min(self._distance(self.snake2.body[0], food) for food in self.food.positions)
        
        # Move snakes
        self.snake1.move(self.ACTION_MAP[action1])
        self.snake2.move(self.ACTION_MAP[action2])
        
        # Initialize rewards
        reward1 = self.step_reward
        reward2 = self.step_reward
        done = False
        
        # Check food consumption
        if self.snake1.body[0] in self.food.positions:
            self.food.remove_food(self.snake1.body[0])
            self.snake1.grow()
            self.scores[0] += 1
            reward1 += self.food_reward
        
        if self.snake2.body[0] in self.food.positions:
            self.food.remove_food(self.snake2.body[0])
            self.snake2.grow()
            self.scores[1] += 1
            reward2 += self.food_reward
        
        # Distance-based rewards
        new_dist1 = min(self._distance(self.snake1.body[0], food) for food in self.food.positions)
        new_dist2 = min(self._distance(self.snake2.body[0], food) for food in self.food.positions)
        
        reward1 += self.approach_reward if new_dist1 < prev_dist1 else self.retreat_penalty
        reward2 += self.approach_reward if new_dist2 < prev_dist2 else self.retreat_penalty
        
        # Competitive element - head-to-head collision
        if self.snake1.body[0] == self.snake2.body[0]:
            reward1 += self.opponent_block_reward
            reward2 += self.opponent_block_reward
        
        # Check collisions
        if self.snake1.check_collision(self.snake2):
            reward1 += self.collide_reward
            done = True
            
        if self.snake2.check_collision(self.snake1):
            reward2 += self.collide_reward
            done = True
            
        return self.get_state(), (reward1, reward2), done, None
    
    def _distance(self, pos1, pos2):
        """Manhattan distance between two points"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self, show_timer=False, elapsed_time=0):
        """Render the current game state without Q-values"""
        self.screen.fill((0, 0, 0))
        
        # Draw game elements
        self.snake1.draw(self.screen)
        self.snake2.draw(self.screen)
        self.food.draw(self.screen)
        
        # Display scores
        score1_text = self.font.render(f"Green: {self.scores[0]}", True, (0, 255, 0))
        score2_text = self.font.render(f"Blue: {self.scores[1]}", True, (0, 0, 255))
        self.screen.blit(score1_text, (20, 10))
        self.screen.blit(score2_text, (self.screen_width - 120, 10))
        
        # Display timer if enabled
        if show_timer:
            remaining = max(0, self.game_time - elapsed_time)
            timer_text = self.font.render(f"Time: {remaining}s", True, (255, 255, 255))
            self.screen.blit(timer_text, (self.screen_width//2 - 50, 10))
        
        pygame.display.flip()
    
    def save_training(self, filename=None):
        """Save the current Q-tables to a file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snake_training_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table1': self.q_table1,
                'q_table2': self.q_table2,
                'scores': self.scores,
                'params': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'food_reward': self.food_reward,
                    'collide_reward': self.collide_reward
                }
            }, f)
        print(f"Training saved to {filename}")
    
    def load_training(self, filename):
        """Load Q-tables from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table1 = data['q_table1']
            self.q_table2 = data['q_table2']
            self.scores = data.get('scores', [0, 0])
            params = data.get('params', {})
            self.alpha = params.get('alpha', 0.2)
            self.gamma = params.get('gamma', 0.95)
        print(f"Training loaded from {filename}")
    
    def update_q_table(self, state, action, reward, next_state, q_table):
        """Update Q-table using Q-learning algorithm"""
        if state not in q_table:
            q_table[state] = [0, 0, 0, 0]
        if next_state not in q_table:
            q_table[next_state] = [0, 0, 0, 0]
            
        current_q = q_table[state][action]
        max_next_q = max(q_table[next_state])
        q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def choose_action(self, state, q_table, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, 3)  # Exploration
        
        # Exploitation with tie-breaking
        q_values = q_table.get(state, [0, 0, 0, 0])
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
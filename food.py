import pygame
import random

class Food:
    def __init__(self, screen_width, screen_height, num_foods=8):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.block_size = 20
        self.positions = []
        self.num_foods = num_foods
        self.generate_foods()
        
    def generate_foods(self):
        """Generate initial food positions"""
        self.positions = []
        for _ in range(self.num_foods):
            self.add_food()
    
    def add_food(self):
        """Add a single food at random position"""
        while True:
            new_pos = (
                random.randint(0, (self.screen_width // self.block_size) - 1) * self.block_size,
                random.randint(0, (self.screen_height // self.block_size) - 1) * self.block_size
            )
            if new_pos not in self.positions:
                self.positions.append(new_pos)
                break
    
    def remove_food(self, position):
        """Remove eaten food and add new one"""
        if position in self.positions:
            self.positions.remove(position)
            self.add_food()
    
    def draw(self, surface):
        """Draw all food items"""
        for pos in self.positions:
            pygame.draw.rect(surface, (255, 0, 0), 
                           pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
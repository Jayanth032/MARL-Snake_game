import pygame

class Snake:
    def __init__(self, start_pos, color, screen_width, screen_height):
        self.body = [start_pos]
        self.direction = (1, 0)  # Initial direction: right
        self.color = color
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.block_size = 20
        self.grow_pending = False

    def move(self, action=None):
        # Update direction if valid move
        if action is not None:
            # Prevent 180-degree turns
            if (action[0] != -self.direction[0] or action[1] != -self.direction[1]):
                self.direction = action

        # Calculate new head position
        head_x, head_y = self.body[0]
        new_head = (
            (head_x + self.direction[0] * self.block_size) % self.screen_width,
            (head_y + self.direction[1] * self.block_size) % self.screen_height
        )

        self.body.insert(0, new_head)
        
        # Remove tail unless growing
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False

    def grow(self):
        self.grow_pending = True

    def draw(self, surface):
        for segment in self.body:
            pygame.draw.rect(
                surface, 
                self.color, 
                pygame.Rect(segment[0], segment[1], self.block_size, self.block_size)
            )
    def check_collision(self, other_snake=None):
        head = self.body[0]
        
        # Check self-collision (skip tail if growing)
        body_to_check = self.body[1:-1] if self.grow_pending else self.body[1:]
        if head in body_to_check:
            return True
        
        # Check collision with other snake
        if other_snake and head in other_snake.body:
            return True
            
        return False
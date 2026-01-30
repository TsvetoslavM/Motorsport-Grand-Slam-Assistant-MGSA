import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MGSA Racing HUD Simulator - Playable Edition")
clock = pygame.time.Clock()

# Colors - Enhanced palette
BLACK = (0, 0, 0)
DARK_GRAY = (15, 15, 18)
GRAY = (45, 45, 50)
WHITE = (255, 255, 255)
NEON_GREEN = (0, 255, 150)
NEON_BLUE = (0, 200, 255)
NEON_RED = (255, 60, 60)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
PURPLE = (180, 100, 255)

# Modern thin font for racing HUD
pygame.font.init()
font_digit = pygame.font.SysFont('arial', 72, bold=False)
font_label = pygame.font.SysFont('arial', 24, bold=False)
font_status = pygame.font.SysFont('arial', 28, bold=False)
font_turn = pygame.font.SysFont('arial', 36, bold=True)
font_small = pygame.font.SysFont('arial', 20, bold=False)

class ParticleEffect:
    """LED motion trail particle system"""
    def __init__(self):
        self.particles = []
    
    def add_particle(self, x, y, color, size):
        self.particles.append({
            'x': x, 'y': y, 'color': color, 
            'size': size, 'alpha': 255, 'life': 1.0
        })
    
    def update(self, dt):
        for p in self.particles[:]:
            p['life'] -= dt * 3
            p['alpha'] = int(255 * p['life'])
            p['size'] *= 0.97
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def draw(self, surface):
        for p in self.particles:
            if p['alpha'] > 0:
                color = tuple(list(p['color'])[:3] + [max(0, p['alpha'])])
                s = pygame.Surface((int(p['size'] * 2), int(p['size'] * 2)), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (int(p['size']), int(p['size'])), int(p['size']))
                surface.blit(s, (int(p['x'] - p['size']), int(p['y'] - p['size'])))

class SteeringWheel:
    """Realistic steering wheel with rotation"""
    def __init__(self):
        self.rotation = 0
        self.target_rotation = 0
        self.x = WIDTH // 2
        self.y = HEIGHT - 180
        self.radius = 140
        self.grip_radius = 18
        
    def update(self, dt, steering_input):
        self.target_rotation = steering_input * 450  # Max 450 degrees rotation
        rot_diff = self.target_rotation - self.rotation
        self.rotation += rot_diff * dt * 8
    
    def draw(self, surface):
        wheel_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        cx, cy = self.x, self.y
        
        # Outer rim shadow
        pygame.draw.circle(wheel_surface, (10, 10, 10, 120), (cx, cy + 5), self.radius + 8)
        
        # Outer rim
        for i in range(20):
            alpha = int(200 - i * 5)
            radius = self.radius + (20 - i)
            color = (25 + i, 25 + i, 30 + i, alpha)
            pygame.draw.circle(wheel_surface, color, (cx, cy), radius, 2)
        
        # Main wheel body
        pygame.draw.circle(wheel_surface, (30, 30, 35), (cx, cy), self.radius)
        pygame.draw.circle(wheel_surface, (45, 45, 50), (cx, cy), self.radius - 5, 3)
        
        # Spokes
        for spoke in range(3):
            angle = math.radians(self.rotation + spoke * 120)
            
            for w in range(12, 0, -1):
                alpha = int(180 + w * 5)
                spoke_color = (50 + w * 3, 50 + w * 3, 55 + w * 3, alpha)
                
                ix = cx + math.cos(angle) * 30
                iy = cy + math.sin(angle) * 30
                ox = cx + math.cos(angle) * (self.radius - 25)
                oy = cy + math.sin(angle) * (self.radius - 25)
                
                pygame.draw.line(wheel_surface, spoke_color, (ix, iy), (ox, oy), w)
        
        # Hand grips
        for grip_angle in [-90, 90]:
            angle = math.radians(self.rotation + grip_angle)
            gx = cx + math.cos(angle) * self.radius
            gy = cy + math.sin(angle) * self.radius
            
            pygame.draw.circle(wheel_surface, (15, 15, 15, 100), (gx + 2, gy + 2), self.grip_radius + 3)
            pygame.draw.circle(wheel_surface, (40, 40, 45), (gx, gy), self.grip_radius)
            
            for i in range(5):
                offset = i * 3 - 6
                line_start = (gx - 8, gy + offset)
                line_end = (gx + 8, gy + offset)
                pygame.draw.line(wheel_surface, (60, 60, 65), line_start, line_end, 1)
            
            pygame.draw.circle(wheel_surface, (70, 70, 80, 150), (gx - 5, gy - 5), 6)
        
        # Center hub
        pygame.draw.circle(wheel_surface, (20, 20, 25), (cx, cy), 35)
        pygame.draw.circle(wheel_surface, (60, 60, 70), (cx, cy), 30)
        pygame.draw.circle(wheel_surface, (40, 40, 45), (cx, cy), 25)
        
        # MGSA logo
        logo_font = pygame.font.SysFont('arial', 16, bold=True)
        logo_text = logo_font.render("MGSA", True, NEON_GREEN)
        logo_rect = logo_text.get_rect(center=(cx, cy))
        wheel_surface.blit(logo_text, logo_rect)
        
        # Rotation indicator
        indicator_angle = math.radians(self.rotation - 90)
        ind_x = cx + math.cos(indicator_angle) * (self.radius - 15)
        ind_y = cy + math.sin(indicator_angle) * (self.radius - 15)
        pygame.draw.circle(wheel_surface, NEON_RED, (ind_x, ind_y), 6)
        pygame.draw.circle(wheel_surface, (255, 150, 150), (ind_x - 2, ind_y - 2), 3)
        
        surface.blit(wheel_surface, (0, 0))

class RacingSimulator:
    def __init__(self):
        self.speed = 0
        self.target_speed = 0
        self.max_speed = 280
        self.distance = 0
        self.track_position = 0
        self.track_position_smooth = 0
        self.led_position = 0
        self.curve_angle = 0
        self.curve_target = 0
        self.time = 0
        self.lap_time = 0
        
        # Player input
        self.steering_input = 0
        self.throttle_input = 0
        self.brake_input = 0
        
        # Camera effects
        self.camera_shake_x = 0
        self.camera_shake_y = 0
        self.camera_tilt = 0
        self.fov_scale = 1.0
        
        # Track parameters
        self.track_width = 350
        self.horizon_y = HEIGHT // 3
        
        # LED HUD parameters
        self.led_count = 24
        self.led_width = 35
        self.led_height = 12
        self.led_spacing = 4
        self.led_y = 180
        
        # Particle system
        self.particles = ParticleEffect()
        
        # Steering wheel
        self.steering_wheel = SteeringWheel()
        
        # Turn prediction
        self.next_turn_direction = 0
        self.next_turn_distance = 0
        
        # Gear simulation
        self.gear = 1
        self.rpm = 1000
        
        # User assistance
        self.show_help = True
        self.help_timer = 8.0
        
        # Performance metrics
        self.best_lap = 999.0
        self.sector_times = [0, 0, 0]
        self.current_sector = 0
        self.last_sector_time = 0
        
        # G-Force simulation
        self.g_force_lat = 0
        self.g_force_long = 0
        
        # Visual effects
        self.speed_lines = []
        
        # Brake/throttle indicators
        self.throttle = 0
        self.brake = 0
    
    def interpolate_color(self, color1, color2, factor):
        factor = max(0, min(1, factor))
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))
    
    def get_deviation_color(self, deviation):
        abs_dev = abs(deviation)
        
        if abs_dev < 0.1:
            return NEON_GREEN
        elif abs_dev < 0.3:
            factor = (abs_dev - 0.1) / 0.2
            return self.interpolate_color(NEON_GREEN, YELLOW, factor)
        elif abs_dev < 0.6:
            factor = (abs_dev - 0.3) / 0.3
            return self.interpolate_color(YELLOW, ORANGE, factor)
        else:
            factor = min(1.0, (abs_dev - 0.6) / 0.4)
            if deviation < 0:
                return self.interpolate_color(ORANGE, NEON_BLUE, factor)
            else:
                return self.interpolate_color(ORANGE, NEON_RED, factor)
    
    def handle_input(self, keys):
        """Handle player keyboard input"""
        # Steering - A/D or LEFT/RIGHT arrows
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.steering_input = max(-1, self.steering_input - 0.08)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.steering_input = min(1, self.steering_input + 0.08)
        else:
            # Auto-center steering
            if abs(self.steering_input) > 0.02:
                self.steering_input *= 0.85
            else:
                self.steering_input = 0
        
        # Throttle - W or UP arrow
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.throttle_input = min(1, self.throttle_input + 0.05)
            self.brake_input = max(0, self.brake_input - 0.1)
        else:
            self.throttle_input = max(0, self.throttle_input - 0.03)
        
        # Brake - S or DOWN arrow
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.brake_input = min(1, self.brake_input + 0.08)
            self.throttle_input = max(0, self.throttle_input - 0.1)
        else:
            self.brake_input = max(0, self.brake_input - 0.05)
    
    def update(self, dt, keys):
        self.time += dt
        self.lap_time += dt
        
        # Handle player input
        self.handle_input(keys)
        
        # Update help timer
        if self.help_timer > 0:
            self.help_timer -= dt
            if self.help_timer <= 0:
                self.show_help = False
        
        # Speed control based on throttle and brake
        acceleration = 80 * self.throttle_input
        braking = 150 * self.brake_input
        drag = 0.5 * self.speed
        
        speed_change = (acceleration - braking - drag) * dt
        self.speed = max(0, min(self.max_speed, self.speed + speed_change))
        
        # Target speed for display
        self.target_speed = self.speed
        
        # Gear and RPM simulation
        if self.speed < 40:
            self.gear = 1
            self.rpm = 1000 + (self.speed / 40) * 5000
        elif self.speed < 80:
            self.gear = 2
            self.rpm = 2000 + ((self.speed - 40) / 40) * 5000
        elif self.speed < 130:
            self.gear = 3
            self.rpm = 2500 + ((self.speed - 80) / 50) * 4500
        elif self.speed < 180:
            self.gear = 4
            self.rpm = 3000 + ((self.speed - 130) / 50) * 4000
        else:
            self.gear = 5
            self.rpm = 3500 + ((self.speed - 180) / 100) * 3500
        
        self.rpm += math.sin(self.time * 30) * 150
        self.rpm = min(8000, self.rpm)
        
        # Dynamic corner generation
        corner_frequency = 0.25
        corner_amplitude = 0.9
        self.curve_target = math.sin(self.time * corner_frequency) * corner_amplitude
        
        # Predict next turn
        future_curve = math.sin((self.time + 3) * corner_frequency) * corner_amplitude
        self.next_turn_direction = 1 if future_curve > 0.2 else (-1 if future_curve < -0.2 else 0)
        self.next_turn_distance = abs(future_curve) * 100
        
        # Ultra-smooth curve transitions
        curve_diff = self.curve_target - self.curve_angle
        self.curve_angle += curve_diff * dt * 1.8
        
        # Player controls car position with steering
        steering_factor = 1.5 if self.speed > 50 else 0.5
        self.track_position += self.steering_input * dt * steering_factor
        
        # Keep car on track (soft boundaries)
        if abs(self.track_position) > 1.2:
            self.track_position *= 0.95
            # Slow down if off track
            self.speed *= 0.98
        
        # Smooth visual position
        smooth_diff = self.track_position - self.track_position_smooth
        self.track_position_smooth += smooth_diff * dt * 4
        
        # LED anticipates movement
        self.led_position = self.track_position_smooth
        
        # Camera shake based on speed and position
        shake_intensity = (self.speed / 100) + abs(self.track_position) * 2
        self.camera_shake_x = math.sin(self.time * 20) * shake_intensity
        self.camera_shake_y = math.cos(self.time * 25) * shake_intensity * 0.5
        
        # Camera tilt based on steering
        target_tilt = -self.steering_input * 1.5
        tilt_diff = target_tilt - self.camera_tilt
        self.camera_tilt += tilt_diff * dt * 3
        
        # Dynamic FOV based on speed
        target_fov = 1.0 + (self.speed / 300) * 0.15
        fov_diff = target_fov - self.fov_scale
        self.fov_scale += fov_diff * dt * 2
        
        # Update distance
        self.distance += self.speed * dt * 0.12
        
        # G-Force calculation
        prev_speed = self.speed - speed_change
        self.g_force_long = (speed_change / dt / 9.81) if dt > 0 else 0
        self.g_force_lat = self.steering_input * (self.speed / 80)
        
        # Update throttle/brake display
        self.throttle = self.throttle_input
        self.brake = self.brake_input
        
        # Sector timing
        sector_length = 30.0
        if self.lap_time >= sector_length * (self.current_sector + 1):
            self.sector_times[self.current_sector] = sector_length
            self.last_sector_time = sector_length
            self.current_sector = (self.current_sector + 1) % 3
            
            if self.current_sector == 0:
                total_lap = sum(self.sector_times)
                if total_lap < self.best_lap:
                    self.best_lap = total_lap
                self.lap_time = 0
        
        # Speed lines effect
        if random.random() < 0.2 + (self.speed / 300):
            self.speed_lines.append({
                'x': random.randint(0, WIDTH),
                'y': random.randint(self.horizon_y, HEIGHT // 2),
                'speed': 3 + self.speed / 15,
                'life': 1.0
            })
        
        for line in self.speed_lines[:]:
            line['y'] += line['speed']
            line['life'] -= dt * 2
            if line['life'] <= 0 or line['y'] > HEIGHT:
                self.speed_lines.remove(line)
        
        # Update systems
        self.particles.update(dt)
        self.steering_wheel.update(dt, self.steering_input)
    
    def draw_vignette(self):
        vignette = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        max_radius = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(0, HEIGHT, 4):
            for x in range(0, WIDTH, 4):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                alpha = int((dist / max_radius) * 80)
                if alpha > 0:
                    pygame.draw.rect(vignette, (0, 0, 0, alpha), (x, y, 4, 4))
        
        # Draw speed lines
        for line in self.speed_lines:
            alpha = int(line['life'] * 100)
            color = (150, 150, 200, alpha)
            pygame.draw.line(vignette, color, 
                           (line['x'], line['y'] - 20), 
                           (line['x'], line['y']), 2)
        
        screen.blit(vignette, (0, 0))
    
    def draw_horizon(self):
        for y in range(self.horizon_y):
            progress = y / self.horizon_y
            color_val = int(15 + progress * 35)
            blue_tint = int(color_val + 5)
            pygame.draw.line(screen, (color_val, color_val, blue_tint), (0, y), (WIDTH, y))
        
        for y in range(self.horizon_y, HEIGHT):
            progress = (y - self.horizon_y) / (HEIGHT - self.horizon_y)
            color_val = int(35 + progress * 25)
            pygame.draw.line(screen, (color_val, color_val, color_val + 2), (0, y), (WIDTH, y))
    
    def draw_track(self):
        num_segments = 40
        
        for i in range(num_segments):
            segment_distance = i / num_segments
            depth = 0.08 + segment_distance * 2.2
            
            future_time = self.time + segment_distance * 4
            future_curve = math.sin(future_time * 0.25) * 0.9
            
            scale = (1 / depth) * self.fov_scale
            y_pos = self.horizon_y + (HEIGHT - self.horizon_y) * segment_distance
            
            tilt_offset = self.camera_tilt * (y_pos - self.horizon_y) * 0.02
            center_offset = -future_curve * 250 * (1 - segment_distance) + self.camera_shake_x
            center_offset += tilt_offset
            
            track_half_width = self.track_width * scale
            
            left_x = WIDTH // 2 - track_half_width + center_offset
            right_x = WIDTH // 2 + track_half_width + center_offset
            
            if i > 0:
                points = [prev_left, prev_right, (right_x, y_pos), (left_x, y_pos)]
                
                base_brightness = 45
                lighting = math.sin(segment_distance * math.pi) * 15
                track_brightness = int(base_brightness + segment_distance * 18 + lighting)
                
                track_color = (track_brightness, track_brightness, track_brightness + 2)
                pygame.draw.polygon(screen, track_color, points)
                
                if segment_distance > 0.1:
                    edge_width = max(1, int(2 * scale))
                    pygame.draw.line(screen, (180, 180, 180), prev_left, (left_x, y_pos), edge_width)
                    pygame.draw.line(screen, (180, 180, 180), prev_right, (right_x, y_pos), edge_width)
                
                if i % 4 == 0 and segment_distance > 0.15:
                    dash_color = (200, 200, 150)
                    dash_width = max(1, int(4 * scale))
                    center_prev = ((prev_left[0] + prev_right[0]) // 2, prev_left[1])
                    center_curr = ((left_x + right_x) // 2, y_pos)
                    pygame.draw.line(screen, dash_color, center_prev, center_curr, dash_width)
                
                # Apex markers
                if i % 10 == 0 and abs(future_curve) > 0.6 and segment_distance > 0.2:
                    marker_x = (left_x + right_x) // 2
                    marker_size = int(10 * scale)
                    if marker_size > 2:
                        pygame.draw.rect(screen, NEON_RED, 
                                       (marker_x - marker_size // 2, y_pos - marker_size // 2, 
                                        marker_size, marker_size))
            
            prev_left = (left_x, y_pos)
            prev_right = (right_x, y_pos)
    
    def draw_ghost_line(self):
        points = []
        num_points = 50
        
        for i in range(num_points):
            segment_distance = i / num_points
            depth = 0.08 + segment_distance * 2.2
            
            future_time = self.time + segment_distance * 4
            future_curve = math.sin(future_time * 0.25) * 0.9
            
            scale = (1 / depth) * self.fov_scale
            y_pos = self.horizon_y + (HEIGHT - self.horizon_y) * segment_distance
            
            ideal_offset = -future_curve * 180 * (1 - segment_distance) + self.camera_shake_x
            tilt_offset = self.camera_tilt * (y_pos - self.horizon_y) * 0.02
            x_pos = WIDTH // 2 + ideal_offset + tilt_offset
            
            points.append((x_pos, y_pos, scale))
        
        if len(points) > 1:
            ghost_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            
            for i in range(len(points) - 1):
                x1, y1, s1 = points[i]
                x2, y2, s2 = points[i + 1]
                
                width = max(1, int(8 * s1))
                alpha = int(120 * (1 - i / len(points)))
                
                glow_color = (0, 255, 150, max(20, alpha // 3))
                pygame.draw.line(ghost_surface, glow_color, (x1, y1), (x2, y2), width + 6)
                
                line_color = (0, 255, 180, alpha)
                pygame.draw.line(ghost_surface, line_color, (x1, y1), (x2, y2), width)
            
            screen.blit(ghost_surface, (0, 0))
    
    def draw_led_hud(self):
        deviation = self.led_position
        
        total_width = self.led_count * (self.led_width + self.led_spacing)
        strip_x = WIDTH // 2 - total_width // 2
        offset_x = -deviation * 180
        
        hud_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        bar_padding = 25
        bar_rect = pygame.Rect(strip_x - bar_padding, self.led_y - bar_padding,
                              total_width + bar_padding * 2, self.led_height + bar_padding * 2)
        
        for i in range(bar_rect.height):
            alpha = int(100 + (i / bar_rect.height) * 50)
            color = (10, 10, 15, alpha)
            pygame.draw.rect(hud_surface, color, 
                           (bar_rect.x, bar_rect.y + i, bar_rect.width, 1))
        
        pygame.draw.rect(hud_surface, (80, 80, 100, 80), bar_rect, 2, border_radius=8)
        
        base_color = self.get_deviation_color(deviation)
        
        for i in range(self.led_count):
            led_x = strip_x + i * (self.led_width + self.led_spacing) + offset_x
            
            center_dist = abs(i - self.led_count / 2) / (self.led_count / 2)
            brightness = 1.0 - center_dist * 0.4
            
            led_color = tuple(int(c * brightness) for c in base_color)
            
            if random.random() < 0.15:
                self.particles.add_particle(led_x + self.led_width // 2, 
                                          self.led_y + self.led_height // 2,
                                          led_color, 4)
            
            bloom_size = 16
            bloom_alpha = int(60 * brightness)
            bloom_color = tuple(list(led_color) + [bloom_alpha])
            bloom_rect = pygame.Rect(led_x - bloom_size // 2, 
                                    self.led_y - bloom_size // 2,
                                    self.led_width + bloom_size, 
                                    self.led_height + bloom_size)
            pygame.draw.rect(hud_surface, bloom_color, bloom_rect, border_radius=8)
            
            led_rect = pygame.Rect(led_x, self.led_y, self.led_width, self.led_height)
            main_alpha = int(240 * brightness)
            main_color = tuple(list(led_color) + [main_alpha])
            pygame.draw.rect(hud_surface, main_color, led_rect, border_radius=4)
            
            highlight_color = tuple(min(255, c + 100) for c in led_color) + (200,)
            highlight_rect = pygame.Rect(led_x + 4, self.led_y + 2, 
                                        self.led_width - 8, 2)
            pygame.draw.rect(hud_surface, highlight_color, highlight_rect, border_radius=1)
        
        self.particles.draw(hud_surface)
        screen.blit(hud_surface, (0, 0))
    
    def draw_cockpit(self):
        cockpit_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        left_pillar = [(0, HEIGHT), (0, 0), (180, 0), (120, HEIGHT)]
        pygame.draw.polygon(cockpit_surface, (5, 5, 8, 200), left_pillar)
        
        right_pillar = [(WIDTH, HEIGHT), (WIDTH, 0), (WIDTH - 180, 0), (WIDTH - 120, HEIGHT)]
        pygame.draw.polygon(cockpit_surface, (5, 5, 8, 200), right_pillar)
        
        top_bar = pygame.Rect(0, 0, WIDTH, 120)
        for i in range(top_bar.height):
            alpha = int(180 - (i / top_bar.height) * 100)
            pygame.draw.rect(cockpit_surface, (3, 3, 5, alpha), (0, i, WIDTH, 1))
        
        screen.blit(cockpit_surface, (0, 0))
    
    def draw_info(self):
        info_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # Speed display
        speed_text = font_digit.render(f"{int(self.speed)}", True, WHITE)
        speed_label = font_label.render("KM/H", True, (180, 180, 180))
        
        speed_x, speed_y = 50, HEIGHT - 320
        
        bg_w, bg_h = 180, 110
        for i in range(bg_h):
            alpha = int(30 + (i / bg_h) * 40)
            pygame.draw.rect(info_surface, (0, 0, 0, alpha), (speed_x - 20, speed_y - 20 + i, bg_w, 1))
        
        info_surface.blit(speed_text, (speed_x, speed_y))
        info_surface.blit(speed_label, (speed_x + 10, speed_y + 60))
        
        # Gear indicator
        gear_text = font_digit.render(f"{self.gear}", True, NEON_GREEN)
        gear_label = font_small.render("GEAR", True, (140, 180, 140))
        info_surface.blit(gear_text, (speed_x + 5, speed_y - 90))
        info_surface.blit(gear_label, (speed_x + 15, speed_y - 110))
        
        # RPM bar
        rpm_max = 8000
        rpm_percent = min(1.0, self.rpm / rpm_max)
        rpm_bar_width = 150
        rpm_bar_height = 12
        
        rpm_x, rpm_y = speed_x, speed_y - 135
        
        pygame.draw.rect(info_surface, (20, 20, 20, 150), 
                        (rpm_x, rpm_y, rpm_bar_width, rpm_bar_height), border_radius=6)
        
        rpm_filled = int(rpm_bar_width * rpm_percent)
        if rpm_percent < 0.7:
            rpm_color = NEON_GREEN
        elif rpm_percent < 0.85:
            rpm_color = YELLOW
        else:
            rpm_color = NEON_RED
        
        pygame.draw.rect(info_surface, rpm_color, 
                        (rpm_x, rpm_y, rpm_filled, rpm_bar_height), border_radius=6)
        
        rpm_text = font_small.render(f"{int(self.rpm)} RPM", True, (180, 180, 180))
        info_surface.blit(rpm_text, (rpm_x, rpm_y - 20))
        
        # Deviation indicator
        deviation_m = self.track_position_smooth * 3.5
        direction = "R" if deviation_m > 0 else "L"
        dev_color = self.get_deviation_color(self.track_position_smooth)
        
        dev_text = font_status.render(f"{abs(deviation_m):.2f}m {direction}", True, dev_color)
        info_surface.blit(dev_text, (speed_x + 5, speed_y + 90))
        
        # Status indicator
        if abs(self.track_position_smooth) < 0.15:
            status = "OPTIMAL"
            status_color = NEON_GREEN
        elif abs(self.track_position_smooth) < 0.4:
            status = "MINOR DEV"
            status_color = YELLOW
        else:
            status = "OFF LINE"
            status_color = NEON_RED
        
        status_text = font_label.render(status, True, status_color)
        info_surface.blit(status_text, (speed_x + 5, speed_y + 120))
        
        # Turn indicator
        if self.next_turn_direction != 0:
            turn_text = "← LEFT" if self.next_turn_direction < 0 else "RIGHT →"
            turn_color = NEON_BLUE if self.next_turn_direction < 0 else NEON_RED
            turn_surface = font_turn.render(turn_text, True, turn_color)
            
            turn_x = WIDTH // 2 - turn_surface.get_width() // 2
            turn_y = 100
            
            pulse = (math.sin(self.time * 4) + 1) / 2
            alpha = int(150 + pulse * 100)
            turn_surface.set_alpha(alpha)
            
            info_surface.blit(turn_surface, (turn_x, turn_y))
        
        # Lap timer
        minutes = int(self.lap_time // 60)
        seconds = int(self.lap_time % 60)
        milliseconds = int((self.lap_time % 1) * 1000)
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        
        timer_text = font_status.render(time_str, True, (180, 220, 255))
        timer_label = font_label.render("LAP TIME", True, (140, 160, 180))
        
        timer_x = WIDTH - 220
        timer_y = HEIGHT - 320
        
        for i in range(70):
            alpha = int(30 + (i / 70) * 40)
            pygame.draw.rect(info_surface, (0, 0, 0, alpha), (timer_x - 15, timer_y - 15 + i, 200, 1))
        
        info_surface.blit(timer_text, (timer_x, timer_y))
        info_surface.blit(timer_label, (timer_x + 10, timer_y + 35))
        
        screen.blit(info_surface, (0, 0))
    
    def draw_g_force_meter(self):
        gf_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        gf_x, gf_y = 50, 50
        gf_size = 100
        
        for i in range(5):
            alpha = 40 - i * 5
            pygame.draw.circle(gf_surface, (0, 0, 0, alpha), 
                             (gf_x + gf_size // 2, gf_y + gf_size // 2), 
                             gf_size // 2 - i)
        
        center_x = gf_x + gf_size // 2
        center_y = gf_y + gf_size // 2
        
        pygame.draw.line(gf_surface, (60, 60, 70, 150), 
                        (gf_x, center_y), (gf_x + gf_size, center_y), 1)
        pygame.draw.line(gf_surface, (60, 60, 70, 150), 
                        (center_x, gf_y), (center_x, gf_y + gf_size), 1)
        
        pygame.draw.circle(gf_surface, (80, 80, 100, 100), 
                         (center_x, center_y), gf_size // 2, 2)
        
        g_lat_px = int(self.g_force_lat * 20)
        g_long_px = int(-self.g_force_long * 20)
        
        dot_x = center_x + g_lat_px
        dot_y = center_y + g_long_px
        
        trail_alpha = 80
        pygame.draw.line(gf_surface, (*NEON_GREEN, trail_alpha), 
                        (center_x, center_y), (dot_x, dot_y), 3)
        
        pygame.draw.circle(gf_surface, NEON_GREEN, (dot_x, dot_y), 6)
        pygame.draw.circle(gf_surface, (200, 255, 200), (dot_x - 2, dot_y - 2), 3)
        
        gf_label = font_small.render("G-FORCE", True, (140, 160, 180))
        gf_surface.blit(gf_label, (gf_x + 5, gf_y + gf_size + 5))
        
        gf_values = font_small.render(f"L:{self.g_force_lat:.1f} F:{self.g_force_long:.1f}", 
                                     True, (180, 180, 180))
        gf_surface.blit(gf_values, (gf_x - 10, gf_y + gf_size + 25))
        
        screen.blit(gf_surface, (0, 0))
    
    def draw_throttle_brake_bars(self):
        tb_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        tb_x = 260
        tb_y = HEIGHT - 280
        bar_width = 30
        bar_height = 150
        bar_spacing = 15
        
        # Throttle bar
        throttle_x = tb_x
        throttle_filled = int(bar_height * self.throttle)
        
        pygame.draw.rect(tb_surface, (20, 40, 20, 150), 
                        (throttle_x, tb_y, bar_width, bar_height), border_radius=5)
        
        if throttle_filled > 0:
            pygame.draw.rect(tb_surface, NEON_GREEN, 
                           (throttle_x, tb_y + bar_height - throttle_filled, 
                            bar_width, throttle_filled), border_radius=5)
        
        pygame.draw.rect(tb_surface, (100, 255, 100, 150), 
                        (throttle_x, tb_y, bar_width, bar_height), 2, border_radius=5)
        
        throttle_label = font_small.render("THR", True, NEON_GREEN)
        tb_surface.blit(throttle_label, (throttle_x - 2, tb_y + bar_height + 5))
        
        # Brake bar
        brake_x = tb_x + bar_width + bar_spacing
        brake_filled = int(bar_height * self.brake)
        
        pygame.draw.rect(tb_surface, (40, 20, 20, 150), 
                        (brake_x, tb_y, bar_width, bar_height), border_radius=5)
        
        if brake_filled > 0:
            pygame.draw.rect(tb_surface, NEON_RED, 
                           (brake_x, tb_y + bar_height - brake_filled, 
                            bar_width, brake_filled), border_radius=5)
        
        pygame.draw.rect(tb_surface, (255, 100, 100, 150), 
                        (brake_x, tb_y, bar_width, bar_height), 2, border_radius=5)
        
        brake_label = font_small.render("BRK", True, NEON_RED)
        tb_surface.blit(brake_label, (brake_x, tb_y + bar_height + 5))
        
        screen.blit(tb_surface, (0, 0))
    
    def draw_sector_times(self):
        sector_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        st_x = WIDTH - 220
        st_y = HEIGHT - 220
        
        for i in range(100):
            alpha = int(30 + (i / 100) * 40)
            pygame.draw.rect(sector_surface, (0, 0, 0, alpha), 
                           (st_x - 15, st_y - 15 + i, 200, 1))
        
        sector_title = font_label.render("SECTORS", True, (140, 160, 180))
        sector_surface.blit(sector_title, (st_x, st_y))
        
        for i in range(3):
            sector_num = f"S{i + 1}"
            
            if i < self.current_sector or self.lap_time > 0:
                time_val = self.sector_times[i]
                color = NEON_GREEN if i == self.current_sector - 1 else (180, 180, 180)
            else:
                time_val = 0
                color = (100, 100, 100)
            
            sector_text = font_small.render(f"{sector_num}: {time_val:.2f}s", True, color)
            sector_surface.blit(sector_text, (st_x + 10, st_y + 30 + i * 25))
        
        if self.best_lap < 999:
            best_label = font_small.render("BEST LAP", True, PURPLE)
            best_time = font_status.render(f"{self.best_lap:.2f}s", True, PURPLE)
            sector_surface.blit(best_label, (st_x + 10, st_y + 110))
            sector_surface.blit(best_time, (st_x + 10, st_y + 130))
        
        screen.blit(sector_surface, (0, 0))
    
    def draw_mini_map(self):
        map_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        map_x, map_y = WIDTH - 200, HEIGHT - 480
        map_width, map_height = 150, 120
        
        for i in range(map_height):
            alpha = int(40 + (i / map_height) * 30)
            pygame.draw.rect(map_surface, (0, 0, 0, alpha), (map_x, map_y + i, map_width, 1))
        
        pygame.draw.rect(map_surface, (60, 60, 80, 100), (map_x, map_y, map_width, map_height), 2, border_radius=8)
        
        map_title = font_small.render("TRACK MAP", True, (140, 160, 180))
        map_surface.blit(map_title, (map_x + 10, map_y + 5))
        
        num_points = 30
        points = []
        
        for i in range(num_points):
            t = self.time + i * 0.3
            curve = math.sin(t * 0.25) * 0.9
            
            x = map_x + map_width // 2 + curve * 40
            y = map_y + 35 + (i / num_points) * (map_height - 45)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(map_surface, (100, 100, 110), False, points, 3)
            pygame.draw.lines(map_surface, NEON_GREEN, False, points, 1)
        
        car_pos = points[3] if len(points) > 3 else (map_x + map_width // 2, map_y + 50)
        pygame.draw.circle(map_surface, NEON_RED, (int(car_pos[0]), int(car_pos[1])), 5)
        pygame.draw.circle(map_surface, (255, 150, 150), (int(car_pos[0]), int(car_pos[1])), 3)
        
        screen.blit(map_surface, (0, 0))
    
    def draw_help_overlay(self):
        if not self.show_help:
            return
        
        help_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        pygame.draw.rect(help_surface, (0, 0, 0, 180), (WIDTH // 2 - 350, 50, 700, 500), border_radius=15)
        pygame.draw.rect(help_surface, NEON_GREEN, (WIDTH // 2 - 350, 50, 700, 500), 3, border_radius=15)
        
        title_font = pygame.font.SysFont('arial', 42, bold=True)
        title = title_font.render("MGSA RACING HUD - PLAYABLE", True, NEON_GREEN)
        title_rect = title.get_rect(center=(WIDTH // 2, 90))
        help_surface.blit(title, title_rect)
        
        help_texts = [
            ("CONTROLS", "", YELLOW),
            ("", "W / UP Arrow    - Accelerate (Throttle)", WHITE),
            ("", "S / DOWN Arrow  - Brake", WHITE),
            ("", "A / LEFT Arrow  - Steer Left", WHITE),
            ("", "D / RIGHT Arrow - Steer Right", WHITE),
            ("", "", WHITE),
            ("HUD ELEMENTS", "", YELLOW),
            ("LED INDICATOR", "Green = Optimal | Yellow/Orange = Minor Dev | Red/Blue = Off Line", NEON_GREEN),
            ("GHOST LINE", "Follow the glowing green line for ideal racing trajectory", NEON_GREEN),
            ("G-FORCE METER", "Green dot shows lateral and longitudinal forces", NEON_GREEN),
            ("", "", WHITE),
            ("", "Press H to toggle help | ESC to exit", (150, 150, 150)),
        ]
        
        y_offset = 140
        for i, (label, desc, color) in enumerate(help_texts):
            if label and desc:
                label_text = font_status.render(label, True, color)
                help_surface.blit(label_text, (WIDTH // 2 - 320, y_offset))
                
                desc_text = font_small.render(desc, True, (200, 200, 200))
                help_surface.blit(desc_text, (WIDTH // 2 - 320, y_offset + 28))
                
                y_offset += 60
            elif label:
                label_text = font_status.render(label, True, color)
                help_surface.blit(label_text, (WIDTH // 2 - 320, y_offset))
                y_offset += 35
            else:
                desc_text = font_small.render(desc, True, color)
                if "Press H" in desc or "Arrow" in desc:
                    help_surface.blit(desc_text, (WIDTH // 2 - 320, y_offset))
                else:
                    desc_rect = desc_text.get_rect(center=(WIDTH // 2, y_offset))
                    help_surface.blit(desc_text, desc_rect)
                y_offset += 25
        
        pulse = (math.sin(self.time * 3) + 1) / 2
        glow_alpha = int(50 + pulse * 100)
        pygame.draw.rect(help_surface, (*NEON_GREEN, glow_alpha), 
                        (WIDTH // 2 - 360, 40, 720, 520), 8, border_radius=20)
        
        screen.blit(help_surface, (0, 0))

def main():
    simulator = RacingSimulator()
    running = True
    
    print("=" * 60)
    print("MGSA RACING HUD SIMULATOR - PLAYABLE EDITION")
    print("=" * 60)
    print("\nFEATURES:")
    print("  • Full player control with keyboard")
    print("  • Professional LED HUD with motion blur and bloom")
    print("  • Realistic steering wheel with rotation indicator")
    print("  • Real-time G-Force meter (lateral & longitudinal)")
    print("  • Throttle and brake pedal indicators")
    print("  • Sector timing and best lap tracking")
    print("  • Smooth camera dynamics (shake, tilt, FOV)")
    print("  • Neon ghost racing line with apex markers")
    print("  • Speed line effects for motion blur")
    print("  • Mini track map with live position")
    print("  • Gear and RPM simulation")
    print("\nCONTROLS:")
    print("  W / UP    - Accelerate (Throttle)")
    print("  S / DOWN  - Brake")
    print("  A / LEFT  - Steer Left")
    print("  D / RIGHT - Steer Right")
    print("  H         - Toggle help overlay")
    print("  ESC       - Exit simulation")
    print("\nHUD ELEMENTS:")
    print("  Top-Left    : G-Force meter")
    print("  Left Side   : Speed, Gear, RPM, Status, Throttle/Brake")
    print("  Top-Center  : LED trajectory indicator")
    print("  Top-Right   : Mini track map")
    print("  Right Side  : Lap timer, Sector times")
    print("  Bottom      : Steering wheel")
    print("\nTIPS:")
    print("  • Follow the green ghost line for optimal trajectory")
    print("  • Brake before corners, accelerate on exit")
    print("  • Keep the LED indicator green for best performance")
    print("  • Watch the G-Force meter to feel the car dynamics")
    print("\n" + "=" * 60)
    print("Starting simulation...\n")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_h:
                    simulator.show_help = not simulator.show_help
                    if simulator.show_help:
                        simulator.help_timer = 999
        
        keys = pygame.key.get_pressed()
        simulator.update(dt, keys)
        
        screen.fill(BLACK)
        simulator.draw_horizon()
        simulator.draw_track()
        simulator.draw_ghost_line()
        simulator.draw_vignette()
        simulator.draw_led_hud()
        simulator.draw_g_force_meter()
        simulator.draw_throttle_brake_bars()
        simulator.draw_cockpit()
        simulator.steering_wheel.draw(screen)
        simulator.draw_info()
        simulator.draw_mini_map()
        simulator.draw_sector_times()
        simulator.draw_help_overlay()
        
        fps_text = font_label.render(f"FPS: {int(clock.get_fps())}", True, (100, 100, 100))
        screen.blit(fps_text, (WIDTH - 100, 10))
        
        brand_text = font_small.render("MGSA © 2025", True, (80, 80, 80))
        screen.blit(brand_text, (WIDTH // 2 - 50, HEIGHT - 25))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
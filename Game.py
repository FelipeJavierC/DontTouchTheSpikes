import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path

# =========================
# Configuración general
# =========================

WIDTH, HEIGHT = 400, 600
FPS = 60

BIRD_HEIGHT = 36  # alto objetivo; el ancho sigue la relación de aspecto
BIRD_START_X = WIDTH // 2
GRAVITY = 0.45
JUMP_FORCE = -7.5
BASE_SPEED = 3.5
SLOW_SPEED = 2.2  # desde score >= 100

# Pinchos: mismo tamaño base en todos los lados, sin solaparse
SIDE_SLOTS = 12
TOP_BOTTOM_COUNT = 9
SPIKE_BASE = WIDTH / TOP_BOTTOM_COUNT          # base del triángulo
_SLOT_H = SPIKE_BASE                          # mismo tamaño en laterales
SPIKE_DEPTH = (HEIGHT - SIDE_SLOTS * _SLOT_H) / 2  # profundidad punta
WALL_MARGIN = 8

_PLAY_TOP = SPIKE_DEPTH
_PLAY_BOTTOM = HEIGHT - SPIKE_DEPTH

PUBLIC_DIR = Path(__file__).resolve().parent / "Public"
BIRD_IMG_PATH = PUBLIC_DIR / "pajaro.png"
MUSIC_PATH = PUBLIC_DIR / "Smooth Criminal - Michael Jackson.mp3"
CANDY_IMG_PATHS = {
    "blue": PUBLIC_DIR / "dulce azul.png",
    "orange": PUBLIC_DIR / "dulce naranjo.png",
    "pink": PUBLIC_DIR / "dulce rosado.png",
}
CANDY_HEIGHT = 24
# dulce más adentro que la punta de los pinchos laterales
CANDY_WALL_INSET = SPIKE_DEPTH + 16

ACTIONS = [0, 1]  # 0 = no aletear, 1 = aletear
STATE_SIZE = 10

# Multiplicadores de velocidad de simulación (0 = sin tope de FPS)
SPEED_STEPS = [0.5, 1, 2, 4, 8, 16, 0]
VOLUME_STEP = 0.1
DEFAULT_VOLUME = 0.3


def side_spike_count(score):
    if score < 5:
        return 2
    if score < 10:
        return 3
    if score < 20:
        return 4
    if score < 35:
        return 5
    if score < 50:
        return 6
    return 7


def candy_tier(score):
    if score < 5:
        return "blue"
    if score < 10:
        return "orange"
    return "pink"


def pick_side_slots(n, score, num_slots=SIDE_SLOTS, forbidden=()):
    """Elige n índices de slot. <20: sin adyacentes. >=20: permite bloques 2–3."""
    forbidden = set(forbidden)
    available = [i for i in range(num_slots) if i not in forbidden]
    if not available:
        available = list(range(num_slots))
    n = min(n, len(available))

    if score < 20:
        # no consecutivos; orden aleatorio (no sorted, o siempre salen abajo)
        chosen = []
        candidates = available[:]
        random.shuffle(candidates)
        for slot in candidates:
            if all(abs(slot - c) >= 2 for c in chosen):
                chosen.append(slot)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            for slot in candidates:
                if slot not in chosen:
                    chosen.append(slot)
                if len(chosen) == n:
                    break
        return sorted(chosen)

    occupied = set()
    remaining = n
    while remaining > 0:
        max_c = min(3, remaining)
        cluster = random.randint(2, max_c) if max_c >= 2 and random.random() < 0.75 else 1
        starts = [
            i
            for i in available
            if i + cluster - 1 < num_slots
            and all(
                (j in available and j not in occupied)
                for j in range(i, i + cluster)
            )
        ]
        if not starts:
            free = [i for i in available if i not in occupied]
            if not free:
                break
            occupied.add(random.choice(free))
            remaining -= 1
            continue
        start = random.choice(starts)
        for j in range(start, start + cluster):
            occupied.add(j)
        remaining -= cluster
    return sorted(occupied)


# =========================
# Red neuronal DQN
# =========================

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Helpers de juego
# =========================

def candy_value(score):
    if score < 5:
        return 1
    if score < 10:
        return 2
    return 3


def bg_color(score):
    if score >= 100:
        return (15, 15, 18)
    if score >= 25:
        # intensos: rosa / naranja / rojo según franja
        band = ((score - 25) // 5) % 3
        return [(255, 180, 200), (255, 200, 150), (255, 160, 150)][band]
    if score >= 20:
        return (235, 220, 245)  # blanco-púrpura
    if score >= 10:
        band = (score - 10) // 5
        return [(210, 235, 190), (235, 235, 170)][min(band, 1)]
    if score >= 5:
        return (200, 220, 240)  # azul suave
    return (245, 242, 235)     # gris/blanco


def spike_color(score):
    if score >= 100:
        return (255, 40, 40)
    return (90, 55, 40)


def triangle_points(cx, tip_y, base_y, half_w):
    return [(cx - half_w, base_y), (cx + half_w, base_y), (cx, tip_y)]


def point_in_triangle(px, py, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri

    def sign(ax, ay, bx, by, cx, cy):
        return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)

    b1 = sign(px, py, x1, y1, x2, y2) < 0
    b2 = sign(px, py, x2, y2, x3, y3) < 0
    b3 = sign(px, py, x3, y3, x1, y1) < 0
    return b1 == b2 == b3


def rect_hits_triangle(rect, tri):
    corners = [
        (rect.left, rect.top),
        (rect.right, rect.top),
        (rect.left, rect.bottom),
        (rect.right, rect.bottom),
        (rect.centerx, rect.centery),
    ]
    if any(point_in_triangle(x, y, tri) for x, y in corners):
        return True
    # sample triangle centroid against rect
    cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3
    cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3
    return rect.collidepoint(cx, cy)


# =========================
# Entorno gráfico
# =========================

class SpikesEnv:
    def __init__(self, render=True):
        self.render_enabled = render
        self.bird_img = None
        self.bird_img_left = None
        self.candy_imgs = {}
        self.bird_w = BIRD_HEIGHT
        self.bird_h = BIRD_HEIGHT
        self.candy_w = int(CANDY_HEIGHT * 1.8)
        self.candy_h = CANDY_HEIGHT
        self.candy_slot = 0
        self.best_score = 0
        self.speed_idx = SPEED_STEPS.index(1)  # 1x por defecto
        self.volume = DEFAULT_VOLUME
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Don't Touch The Spikes - DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 48)
            self.ui_font = pygame.font.SysFont(None, 24)
            self._load_bird_sprite()
            self._load_candy_sprites()
            pygame.mixer.music.load(str(MUSIC_PATH))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play(-1)  # bucle infinito
            # botones velocidad: círculos centrados, encima de pinchos inferiores
            self.btn_color = (255, 52, 100)  # #ff3464
            d, gap, label_w = 26, 8, 36
            total_w = d + gap + label_w + gap + d
            x0 = (WIDTH - total_w) // 2
            cy = int(HEIGHT - SPIKE_DEPTH - d - 22)
            self.btn_minus = pygame.Rect(x0, cy, d, d)
            self.speed_label_rect = pygame.Rect(x0 + d + gap, cy, label_w, d)
            self.btn_plus = pygame.Rect(x0 + d + gap + label_w + gap, cy, d, d)
        self.reset()

    def sim_speed(self):
        return SPEED_STEPS[self.speed_idx]

    def _change_speed(self, delta):
        self.speed_idx = max(0, min(len(SPEED_STEPS) - 1, self.speed_idx + delta))

    def _change_volume(self, delta):
        self.volume = round(max(0.0, min(1.0, self.volume + delta)), 1)
        if self.render_enabled:
            pygame.mixer.music.set_volume(self.volume)

    def _load_bird_sprite(self):
        img = pygame.image.load(str(BIRD_IMG_PATH)).convert_alpha()
        ow, oh = img.get_size()
        scale = BIRD_HEIGHT / oh
        self.bird_w = max(1, int(round(ow * scale)))
        self.bird_h = max(1, int(round(oh * scale)))
        self.bird_img = pygame.transform.smoothscale(img, (self.bird_w, self.bird_h))
        self.bird_img_left = pygame.transform.flip(self.bird_img, True, False)

    def _load_candy_sprites(self):
        for key, path in CANDY_IMG_PATHS.items():
            img = pygame.image.load(str(path)).convert_alpha()
            ow, oh = img.get_size()
            scale = CANDY_HEIGHT / oh
            w = max(1, int(round(ow * scale)))
            h = max(1, int(round(oh * scale)))
            self.candy_imgs[key] = pygame.transform.smoothscale(img, (w, h))
        sample = self.candy_imgs["blue"]
        self.candy_w, self.candy_h = sample.get_width(), sample.get_height()

    def _speed(self):
        return SLOW_SPEED if self.score >= 100 else BASE_SPEED

    def _playable_top(self):
        return _PLAY_TOP

    def _playable_bottom(self):
        return _PLAY_BOTTOM

    def _slot_center_y(self, slot_i):
        return _PLAY_TOP + (slot_i + 0.5) * _SLOT_H

    def _candy_forbidden_slots(self):
        """Slots que no deben llevar pincho si el dulce está en esa pared."""
        if not getattr(self, "candy_alive", False):
            return set()
        if getattr(self, "candy_side", 0) != getattr(self, "target_wall", 0):
            return set()
        slot = getattr(self, "candy_slot", None)
        if slot is None:
            return set()
        # slot del dulce + vecinos para margen visual
        return {s for s in (slot - 1, slot, slot + 1) if 0 <= s < SIDE_SLOTS}

    def _regen_side_spikes(self):
        """Pinchos en la pared hacia la que vuela el pájaro (al rebotar en la opuesta)."""
        self.target_wall = 1 if self.vx > 0 else -1
        n = side_spike_count(self.score)
        forbidden = self._candy_forbidden_slots()
        self.side_slots = pick_side_slots(n, self.score, forbidden=forbidden)
        self.side_spikes = [self._slot_center_y(i) for i in self.side_slots]

    def _spawn_candy(self):
        # posición fija hasta que se coma; sin solapar pinchos laterales
        self.candy_side = random.choice([-1, 1])
        blocked = set(self.side_slots) if self.candy_side == self.target_wall else set()
        free = [
            i
            for i in range(SIDE_SLOTS)
            if all(abs(i - s) >= 2 for s in blocked)
        ]
        if not free:
            free = [i for i in range(SIDE_SLOTS) if i not in blocked] or list(range(SIDE_SLOTS))
        self.candy_slot = random.choice(free)
        self.candy_y = self._slot_center_y(self.candy_slot)
        self.candy_alive = True

    def reset(self):
        self.bird_x = float(BIRD_START_X)
        self.bird_y = float(HEIGHT // 2)
        self.vy = 0.0
        self.vx = BASE_SPEED
        self.score = 0
        self.done = False
        self.side_slots = []
        self._regen_side_spikes()
        self._spawn_candy()
        return self.get_state()

    def _nearest_side_spike_y(self):
        if not self.side_spikes:
            return 0.5
        best = min(self.side_spikes, key=lambda y: abs(y - self.bird_y))
        return best / HEIGHT

    def _gap_center_norm(self):
        """Centro del hueco más grande entre pinchos laterales (aprox. zona segura)."""
        occupied = set(self.side_slots)
        best_gap = 0
        best_mid = HEIGHT / 2
        run_start = None
        for i in range(SIDE_SLOTS + 1):
            free = i < SIDE_SLOTS and i not in occupied
            if free and run_start is None:
                run_start = i
            if (not free or i == SIDE_SLOTS) and run_start is not None:
                end = i
                gap = (end - run_start) * _SLOT_H
                mid = _PLAY_TOP + (run_start + end) / 2 * _SLOT_H
                if gap > best_gap:
                    best_gap = gap
                    best_mid = mid
                run_start = None
        return best_mid / HEIGHT, best_gap / HEIGHT

    def get_state(self):
        gap_y, gap_size = self._gap_center_norm()
        candy_y = self.candy_y / HEIGHT if self.candy_alive else -1.0
        candy_side = float(self.candy_side) if self.candy_alive else 0.0
        return np.array(
            [
                self.bird_y / HEIGHT,
                self.vy / 10.0,
                self.bird_x / WIDTH,
                1.0 if self.vx > 0 else -1.0,
                self._nearest_side_spike_y(),
                gap_y,
                gap_size,
                candy_y,
                candy_side,
                min(self.score, 150) / 150.0,
            ],
            dtype=np.float32,
        )

    def _bird_rect(self):
        return pygame.Rect(
            int(self.bird_x - self.bird_w // 2),
            int(self.bird_y - self.bird_h // 2),
            self.bird_w,
            self.bird_h,
        )

    def _top_bottom_tris(self):
        tris = []
        half = SPIKE_BASE / 2
        for i in range(TOP_BOTTOM_COUNT):
            cx = i * SPIKE_BASE + half
            tris.append(triangle_points(cx, SPIKE_DEPTH, 0, half))
            tris.append(triangle_points(cx, HEIGHT - SPIKE_DEPTH, HEIGHT, half))
        return tris

    def _side_tris(self):
        tris = []
        half = _SLOT_H / 2
        for slot_i in self.side_slots:
            cy = self._slot_center_y(slot_i)
            if self.target_wall > 0:
                tris.append(
                    [
                        (WIDTH, cy - half),
                        (WIDTH, cy + half),
                        (WIDTH - SPIKE_DEPTH, cy),
                    ]
                )
            else:
                tris.append(
                    [
                        (0, cy - half),
                        (0, cy + half),
                        (SPIKE_DEPTH, cy),
                    ]
                )
        return tris

    def _candy_rect(self):
        if self.candy_side < 0:
            cx = CANDY_WALL_INSET + self.candy_w // 2
        else:
            cx = WIDTH - CANDY_WALL_INSET - self.candy_w // 2
        return pygame.Rect(
            int(cx - self.candy_w // 2),
            int(self.candy_y - self.candy_h // 2),
            self.candy_w,
            self.candy_h,
        )

    def _hits_spikes(self, bird):
        for tri in self._top_bottom_tris() + self._side_tris():
            if rect_hits_triangle(bird, tri):
                return True
        if bird.top < SPIKE_DEPTH - 2 or bird.bottom > HEIGHT - SPIKE_DEPTH + 2:
            return True
        return False

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0, True

        reward = 0.01
        speed = self._speed()
        self.vx = speed if self.vx > 0 else -speed

        if action == 1:
            self.vy = JUMP_FORCE

        self.vy += GRAVITY
        self.bird_y += self.vy
        self.bird_x += self.vx

        bird = self._bird_rect()

        # rebote en paredes laterales
        bounced = False
        if bird.right >= WIDTH - WALL_MARGIN:
            self.bird_x = WIDTH - WALL_MARGIN - self.bird_w // 2
            self.vx = -abs(self._speed())
            bounced = True
        elif bird.left <= WALL_MARGIN:
            self.bird_x = WALL_MARGIN + self.bird_w // 2
            self.vx = abs(self._speed())
            bounced = True

        if bounced:
            self.score += 1
            reward += 1.0
            self.best_score = max(self.best_score, self.score)
            self._regen_side_spikes()
            bird = self._bird_rect()

        # dulce: no se mueve hasta comerlo; al comerlo aparece otro
        if self.candy_alive and bird.colliderect(self._candy_rect()):
            val = candy_value(self.score)
            self.score += val
            reward += float(val)
            self.best_score = max(self.best_score, self.score)
            self._spawn_candy()

        if self._hits_spikes(bird):
            reward = -10.0
            self.done = True

        if self.render_enabled:
            self.render()

        return self.get_state(), reward, self.done

    def _draw_speed_buttons(self):
        for rect, label in ((self.btn_minus, "−"), (self.btn_plus, "+")):
            pygame.draw.circle(self.screen, self.btn_color, rect.center, rect.width // 2)
            txt = self.ui_font.render(label, True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        mult = self.sim_speed()
        speed_txt = "MAX" if mult == 0 else f"{mult:g}x"
        label = self.ui_font.render(speed_txt, True, (40, 40, 40))
        self.screen.blit(label, label.get_rect(center=self.speed_label_rect.center))

    def render(self):
        mult = self.sim_speed()
        self.clock.tick(0 if mult == 0 else max(1, int(FPS * mult)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_minus.collidepoint(event.pos):
                    self._change_speed(-1)
                elif self.btn_plus.collidepoint(event.pos):
                    self._change_speed(+1)
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._change_speed(-1)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._change_speed(+1)
                elif event.key == pygame.K_DOWN:
                    self._change_volume(-VOLUME_STEP)
                elif event.key == pygame.K_UP:
                    self._change_volume(+VOLUME_STEP)

        bg = bg_color(self.score)
        sc = spike_color(self.score)
        self.screen.fill(bg)

        # círculo de score (estilo del juego)
        pygame.draw.circle(self.screen, (255, 255, 255), (WIDTH // 2, HEIGHT // 2), 90)
        score_surf = self.font.render(str(self.score), True, (200, 200, 200))
        self.screen.blit(
            score_surf,
            score_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2)),
        )

        for tri in self._top_bottom_tris():
            pygame.draw.polygon(self.screen, sc, tri)
        for tri in self._side_tris():
            pygame.draw.polygon(self.screen, sc, tri)

        # BEST SCORE centrado bajo la pared superior
        best_surf = self.ui_font.render(f"BEST SCORE: {self.best_score}", True, (80, 80, 80))
        self.screen.blit(
            best_surf,
            best_surf.get_rect(center=(WIDTH // 2, int(SPIKE_DEPTH + 30))),
        )

        if self.candy_alive:
            candy_sprite = self.candy_imgs[candy_tier(self.score)]
            self.screen.blit(candy_sprite, candy_sprite.get_rect(center=self._candy_rect().center))

        sprite = self.bird_img if self.vx > 0 else self.bird_img_left
        rect = sprite.get_rect(center=(int(self.bird_x), int(self.bird_y)))
        self.screen.blit(sprite, rect)

        self._draw_speed_buttons()
        pygame.display.flip()

    def close(self):
        if self.render_enabled:
            pygame.mixer.music.stop()
            pygame.quit()


# =========================
# Entrenamiento DQN
# =========================

def train_dqn(
    render=True,
    episodes=10000,
    lr=0.001,
    gamma=0.99,
    epsilon_decay=0.995,
    epsilon_min=0.05,
    batch_size=64,
    target_update=20,
    memory_size=50000,
    max_steps=2000,
    verbose=True,
):
    env = SpikesEnv(render=render)
    model = DQN(STATE_SIZE, len(ACTIONS))
    target_model = DQN(STATE_SIZE, len(ACTIONS))
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=memory_size)
    epsilon = 1.0

    rewards_history = []
    scores_history = []

    def choose_action(state):
        nonlocal epsilon
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        with torch.no_grad():
            q = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q).item())

    def train_step():
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        q_values = model(states)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next_q = torch.max(target_model(next_states), dim=1)[0]
            target = rewards_t + gamma * max_next_q * (1 - dones_t)
        loss = loss_fn(q_value, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action = choose_action(state)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward
            train_step()
            steps += 1
            if done or steps >= max_steps:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        rewards_history.append(total_reward)
        scores_history.append(env.score)
        if verbose and episode % max(1, episodes // 20) == 0:
            print(
                f"Episode: {episode}, Score: {env.score}, "
                f"Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}"
            )

    if render:
        env.close()
    return {
        "model": model,
        "rewards": rewards_history,
        "scores": scores_history,
        "epsilon": epsilon,
    }


if __name__ == "__main__":
    train_dqn(render=True, episodes=10000, verbose=True)

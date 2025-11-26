import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame


@dataclass
class Fruit:
    kind: str
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    rotation: float
    spin: float

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 520 * dt
        self.rotation = (self.rotation + self.spin * dt) % 360


@dataclass
class SliceParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        self.vy += 220 * dt

    def draw(self, surface: pygame.Surface) -> None:
        if self.life <= 0:
            return
        if not hasattr(self, "_sprite"):
            SliceParticle._sprite = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(SliceParticle._sprite, (255, 255, 255), (4, 4), 4)
        alpha = max(30, min(255, int(255 * (self.life / 0.5))))
        SliceParticle._sprite.set_alpha(alpha)
        surface.blit(SliceParticle._sprite, (self.x, self.y), special_flags=pygame.BLEND_ADD)


@dataclass
class FloatingText:
    text: str
    x: float
    y: float
    life: float
    color: tuple

    def update(self, dt: float) -> None:
        self.y -= 30 * dt
        self.life -= dt

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if self.life <= 0:
            return
        alpha = max(40, min(255, int(255 * (self.life / 0.8))))
        render = font.render(self.text, True, self.color)
        render.set_alpha(alpha)
        surface.blit(render, (self.x, self.y))


class ScoreBoard:
    def __init__(self, path: Path, keep_top: int = 5) -> None:
        self.path = path
        self.keep_top = keep_top
        self.scores = self._load_scores()

    def _load_scores(self) -> list[int]:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return list(sorted(data))[-self.keep_top :][::-1]
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def add_score(self, score: int) -> None:
        self.scores.append(score)
        self.scores = sorted(self.scores, reverse=True)[: self.keep_top]
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.scores, f)
        except OSError:
            pass


class HandTracker:
    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        landmark = result.multi_hand_landmarks[0].landmark[8]
        return landmark.x, landmark.y


class FruitSlashGame:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Chém Hoa Quả")
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        try:
            self.screen = pygame.display.set_mode((1100, 720), flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode((1100, 720), flags)
        self.clock = pygame.time.Clock()
        self.play_width = 820
        self.width, self.height = self.screen.get_size()

        self.font = self._load_font(26)
        self.big_font = self._load_font(48, bold=True)

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Camera not available. Please connect a webcam.")
            sys.exit(1)

        self.tracker = HandTracker()
        self.scoreboard = ScoreBoard(Path("slash_scores.json"))
        self.last_frame: np.ndarray | None = None
        self.slice_trail: list[tuple[int, int, float]] = []

        self._build_surfaces()
        self.reset()

    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        candidates = ["DejaVuSans", "Arial", "Helvetica", "LiberationSans", "sans-serif"]
        for name in candidates:
            matched = pygame.font.match_font(name, bold=bold)
            if matched:
                return pygame.font.Font(matched, size)
        return pygame.font.Font(None, size)

    def _build_surfaces(self) -> None:
        self.background = pygame.Surface((self.play_width, self.height)).convert()
        for y in range(self.height):
            t = y / self.height
            color = (int(18 + 30 * t), int(26 + 30 * t), int(34 + 34 * t))
            pygame.draw.line(self.background, color, (0, y), (self.play_width, y))

        self.fruit_surface = self._make_item_surface((120, 220, 120), (190, 255, 190), 24)
        self.bomb_surface = self._make_item_surface((220, 90, 90), (255, 180, 180), 26)
        self.trail_surface = pygame.Surface((self.play_width, self.height), pygame.SRCALPHA)

    def _make_item_surface(self, base: tuple[int, int, int], highlight: tuple[int, int, int], radius: int) -> pygame.Surface:
        size = radius * 2 + 6
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        for r in range(radius, 0, -1):
            t = r / radius
            color = (
                int(base[0] * t + highlight[0] * (1 - t)),
                int(base[1] * t + highlight[1] * (1 - t)),
                int(base[2] * t + highlight[2] * (1 - t)),
            )
            pygame.draw.circle(surf, color, (center, center), r)
        pygame.draw.circle(surf, (255, 255, 255, 90), (center - radius // 3, center - radius // 3), radius // 2)
        return surf.convert_alpha()

    def reset(self) -> None:
        self.fruits: list[Fruit] = []
        self.particles: list[SliceParticle] = []
        self.texts: list[FloatingText] = []
        self.score = 0
        self.misses = 0
        self.last_spawn = 0.0
        self.start_time = time.time()
        self.game_over = False
        self.prev_tip: tuple[float, float] | None = None

    def spawn_fruit(self, level: int) -> None:
        kind = "fruit" if random.random() > 0.15 else "bomb"
        radius = 24 if kind == "fruit" else 26
        x = random.randint(80, self.play_width - 80)
        vy = random.uniform(-600, -520)
        vx = random.uniform(-140, 140)
        spin = random.uniform(-220, 220)
        self.fruits.append(Fruit(kind, x, self.height + 30, vx, vy - level * 15, radius, 0, spin))

    def update_slice_trail(self, tip_px: tuple[int, int] | None, dt: float) -> None:
        now = time.time()
        if tip_px:
            self.slice_trail.append((tip_px[0], tip_px[1], now))
        self.slice_trail = [p for p in self.slice_trail if now - p[2] < 0.2]

    def check_slice(self, tip_norm: tuple[float, float] | None, dt: float) -> None:
        if not tip_norm or not self.prev_tip:
            self.prev_tip = tip_norm
            return
        p1 = (int(self.prev_tip[0] * self.play_width), int(self.prev_tip[1] * self.height))
        p2 = (int(tip_norm[0] * self.play_width), int(tip_norm[1] * self.height))
        distance = math.dist(p1, p2)
        speed = distance / dt if dt > 0 else 0
        self.prev_tip = tip_norm
        if speed < 420:
            return

        for fruit in self.fruits[:]:
            if self._segment_hits_circle(p1, p2, (fruit.x, fruit.y), fruit.radius + 6):
                self.fruits.remove(fruit)
                self.slice_particles(fruit)
                if fruit.kind == "fruit":
                    self.score += 1
                    self.texts.append(FloatingText("+1", fruit.x, fruit.y, 0.8, (200, 255, 200)))
                else:
                    self.misses += 1
                    self.texts.append(FloatingText("BÙM!", fruit.x, fruit.y, 0.8, (255, 180, 180)))
                    if self.misses >= 3:
                        self.game_over = True
                        self.scoreboard.add_score(self.score)

    def _segment_hits_circle(self, p1: tuple[int, int], p2: tuple[int, int], c: tuple[float, float], r: float) -> bool:
        (x1, y1), (x2, y2) = p1, p2
        cx, cy = c
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            return math.hypot(cx - x1, cy - y1) <= r
        t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
        closest = (x1 + t * dx, y1 + t * dy)
        return math.dist(closest, (cx, cy)) <= r

    def slice_particles(self, fruit: Fruit) -> None:
        base_color = (200, 255, 200) if fruit.kind == "fruit" else (255, 140, 140)
        for _ in range(28):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(180, 360)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append(SliceParticle(fruit.x, fruit.y, vx, vy, 0.5, base_color))

    def update_effects(self, dt: float) -> None:
        for particle in self.particles[:]:
            particle.update(dt)
            if particle.life <= 0:
                self.particles.remove(particle)
        for text in self.texts[:]:
            text.update(dt)
            if text.life <= 0:
                self.texts.remove(text)

    def draw_background(self) -> None:
        self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(self.screen, (45, 55, 75), (self.play_width, 0, self.width - self.play_width, self.height))
        pygame.draw.rect(self.screen, (70, 90, 110), (self.play_width, 0, self.width - self.play_width, self.height), 3)

    def draw_fruits(self) -> None:
        for fruit in self.fruits:
            surf = self.fruit_surface if fruit.kind == "fruit" else self.bomb_surface
            rotated = pygame.transform.rotozoom(surf, fruit.rotation, 1.0)
            rect = rotated.get_rect(center=(int(fruit.x), int(fruit.y)))
            self.screen.blit(rotated, rect)

        for particle in self.particles:
            particle.draw(self.screen)

        for text in self.texts:
            text.draw(self.screen, self.font)

    def draw_slice_trail(self) -> None:
        self.trail_surface.fill((0, 0, 0, 0))
        if len(self.slice_trail) > 1:
            points = [(p[0], p[1]) for p in self.slice_trail]
            pygame.draw.lines(self.trail_surface, (120, 220, 255, 140), False, points, 4)
            pygame.draw.lines(self.trail_surface, (255, 255, 255, 220), False, points, 2)
        self.screen.blit(self.trail_surface, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)

    def draw_sidebar(self, tip_norm: tuple[float, float] | None) -> None:
        x = self.play_width + 16
        y = 24
        self.screen.blit(self.big_font.render("Chém Hoa Quả", True, (235, 235, 245)), (x, y))
        y += 80
        self.screen.blit(self.font.render(f"Điểm: {self.score}", True, (220, 230, 240)), (x, y))
        y += 40
        self.screen.blit(self.font.render(f"Bom trúng: {self.misses}/3", True, (255, 180, 180)), (x, y))
        y += 40
        elapsed = time.time() - self.start_time
        level = int(elapsed // 12) + 1
        self.screen.blit(self.font.render(f"Tốc độ: cấp {level}", True, (190, 210, 255)), (x, y))
        y += 50

        self.screen.blit(self.font.render("Top điểm:", True, (230, 230, 240)), (x, y))
        for i, s in enumerate(self.scoreboard.scores, 1):
            y += 28
            self.screen.blit(self.font.render(f"{i}. {s}", True, (210, 210, 220)), (x, y))

        if self.last_frame is not None:
            cam_w, cam_h = 240, 180
            frame = cv2.resize(self.last_frame, (cam_w, cam_h))
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if tip_norm:
                px = int(tip_norm[0] * cam_w)
                py = int(tip_norm[1] * cam_h)
                cv2.circle(display, (px, py), 6, (40, 230, 90), 2)
                cv2.circle(display, (px, py), 3, (30, 200, 60), -1)
            surface = pygame.image.frombuffer(display.tobytes(), (cam_w, cam_h), "RGB")
            cam_x = self.play_width + 10
            cam_y = self.height - cam_h - 24
            pygame.draw.rect(self.screen, (32, 36, 48), (cam_x - 4, cam_y - 4, cam_w + 8, cam_h + 8), 2)
            label = self.font.render("Camera", True, (210, 210, 220))
            self.screen.blit(label, (cam_x, cam_y - 28))
            self.screen.blit(surface, (cam_x, cam_y))

    def draw_game_over(self) -> None:
        if not self.game_over:
            return
        overlay = pygame.Surface((self.play_width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        text = self.big_font.render("Thua rồi!", True, (255, 210, 210))
        score_text = self.font.render(f"Điểm: {self.score}", True, (230, 230, 240))
        hint = self.font.render("Nhấn R để chơi lại hoặc ESC để thoát", True, (230, 230, 240))
        self.screen.blit(text, (self.play_width // 2 - text.get_width() // 2, 240))
        self.screen.blit(score_text, (self.play_width // 2 - score_text.get_width() // 2, 300))
        self.screen.blit(hint, (self.play_width // 2 - hint.get_width() // 2, 350))

    def run(self) -> None:
        while True:
            dt = min(self.clock.tick(120) / 1000.0, 1 / 30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.capture.release()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.capture.release()
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_r and self.game_over:
                        self.reset()

            ret, frame = self.capture.read()
            if not ret:
                continue
            self.last_frame = frame
            tip_norm = self.tracker.detect(frame)
            self.check_slice(tip_norm, dt)
            self.update_slice_trail(
                (int(tip_norm[0] * self.play_width), int(tip_norm[1] * self.height)) if tip_norm else None,
                dt,
            )

            elapsed = time.time() - self.start_time
            level = int(elapsed // 12)
            spawn_interval = max(0.4, 1.0 - level * 0.08)
            self.last_spawn += dt
            if not self.game_over and self.last_spawn >= spawn_interval:
                self.spawn_fruit(level)
                self.last_spawn = 0.0

            if not self.game_over:
                for fruit in self.fruits[:]:
                    fruit.update(dt)
                    if fruit.y > self.height + 80 and fruit.kind == "fruit":
                        self.fruits.remove(fruit)
                        self.misses += 1
                        if self.misses >= 3:
                            self.game_over = True
                            self.scoreboard.add_score(self.score)
                if self.misses >= 3:
                    self.game_over = True
                    self.scoreboard.add_score(self.score)

            self.update_effects(dt)
            self.draw_background()
            self.draw_fruits()
            self.draw_slice_trail()
            self.draw_sidebar(tip_norm)
            self.draw_game_over()
            pygame.display.flip()


if __name__ == "__main__":
    game = FruitSlashGame()
    game.run()

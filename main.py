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
class FallingItem:
    kind: str
    x: float
    y: float
    speed: float
    radius: int

    def update(self, dt: float) -> None:
        self.y += self.speed * dt

    def draw(self, surface: pygame.Surface) -> None:
        color = (248, 196, 43) if self.kind == "coin" else (200, 48, 48)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)
        outline = (255, 235, 128) if self.kind == "coin" else (250, 128, 128)
        pygame.draw.circle(surface, outline, (int(self.x), int(self.y)), self.radius, 2)


@dataclass
class Particle:
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

    def draw(self, surface: pygame.Surface) -> None:
        if self.life > 0:
            alpha = max(30, min(255, int(255 * (self.life / 0.6))))
            s = pygame.Surface((6, 6), pygame.SRCALPHA)
            s.fill((*self.color, alpha))
            surface.blit(s, (self.x, self.y))


@dataclass
class FloatingText:
    text: str
    x: float
    y: float
    life: float
    color: tuple

    def update(self, dt: float) -> None:
        self.y -= 20 * dt
        self.life -= dt

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if self.life > 0:
            alpha = max(40, min(255, int(255 * (self.life / 1.0))))
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


class NoseTracker:
    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmark = result.multi_face_landmarks[0].landmark[1]
        return landmark.x, landmark.y


class MineCartGame:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Mine Cart Camera Collector")
        self.clock = pygame.time.Clock()
        self.width, self.height = 900, 600
        self.play_width = 650
        self.screen = pygame.display.set_mode((self.width, self.height))
        try:
            self.font = pygame.font.SysFont("arial", 24)
            self.big_font = pygame.font.SysFont("arial", 48, bold=True)
        except Exception:
            self.font = pygame.font.Font(None, 24)
            self.big_font = pygame.font.Font(None, 48)

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Camera not available. Please connect a webcam.")
            sys.exit(1)

        self.tracker = NoseTracker()
        self.scoreboard = ScoreBoard(Path("scores.json"))
        self.last_frame: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        self.player_rect = pygame.Rect(self.play_width // 2 - 40, self.height - 60, 80, 30)
        self.items: list[FallingItem] = []
        self.particles: list[Particle] = []
        self.texts: list[FloatingText] = []
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False
        self.start_time = time.time()
        self.last_spawn = 0.0
        self.flash_timer = 0.0

    def spawn_item(self, speed_factor: float) -> None:
        kind = "coin" if random.random() < 0.7 else "bomb"
        x = random.randint(30, self.play_width - 30)
        radius = 14 if kind == "coin" else 16
        base_speed = 140 if kind == "coin" else 170
        speed = base_speed * speed_factor
        self.items.append(FallingItem(kind, x, -20, speed, radius))

    def update_player(self, nose_pos: tuple[float, float] | None) -> None:
        if nose_pos:
            nx = nose_pos[0]
            target_x = int(nx * self.play_width)
            self.player_rect.centerx = max(40, min(self.play_width - 40, target_x))

    def handle_collisions(self) -> None:
        for item in self.items[:]:
            if self.player_rect.collidepoint(item.x, item.y):
                self.items.remove(item)
                if item.kind == "coin":
                    self.score += 1
                    self.spawn_effect(item, (248, 196, 43))
                    self.texts.append(FloatingText("+1", item.x, item.y, 1.0, (248, 196, 43)))
                else:
                    self.bomb_hits += 1
                    self.flash_timer = 0.3
                    self.spawn_effect(item, (230, 76, 60))
                    self.texts.append(FloatingText("BÙM!", item.x, item.y, 1.0, (230, 76, 60)))
                    if self.bomb_hits >= 3:
                        self.game_over = True
                        self.scoreboard.add_score(self.score)

    def spawn_effect(self, item: FallingItem, color: tuple) -> None:
        for _ in range(16):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(80, 180)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append(Particle(item.x, item.y, vx, vy, 0.6, color))

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
        self.screen.fill((18, 18, 28))
        pygame.draw.rect(self.screen, (30, 30, 44), (0, 0, self.play_width, self.height))
        pygame.draw.rect(self.screen, (45, 45, 55), self.player_rect.inflate(8, 8), 2, border_radius=6)

    def draw_scoreboard(self) -> None:
        panel_rect = pygame.Rect(self.play_width, 0, self.width - self.play_width, self.height)
        pygame.draw.rect(self.screen, (22, 26, 38), panel_rect)
        title = self.big_font.render("Bảng điểm", True, (235, 235, 245))
        self.screen.blit(title, (self.play_width + 20, 20))

        score_text = self.font.render(f"Điểm hiện tại: {self.score}", True, (220, 220, 230))
        self.screen.blit(score_text, (self.play_width + 20, 100))

        bomb_text = self.font.render(f"Bom trúng: {self.bomb_hits}/3", True, (235, 140, 140))
        self.screen.blit(bomb_text, (self.play_width + 20, 140))

        elapsed = time.time() - self.start_time
        level = int(elapsed // 15)
        level_text = self.font.render(f"Tốc độ: cấp {level + 1}", True, (180, 200, 255))
        self.screen.blit(level_text, (self.play_width + 20, 180))

        self.screen.blit(self.font.render("Top điểm:", True, (230, 230, 240)), (self.play_width + 20, 240))
        for i, score in enumerate(self.scoreboard.scores, 1):
            txt = self.font.render(f"{i}. {score}", True, (210, 210, 220))
            self.screen.blit(txt, (self.play_width + 20, 240 + i * 28))

    def draw_camera_view(self, nose_pos: tuple[float, float] | None) -> None:
        if self.last_frame is None:
            return

        view_width = self.width - self.play_width - 40
        view_height = 180
        frame = cv2.resize(self.last_frame, (view_width, view_height))
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if nose_pos:
            px = int(nose_pos[0] * view_width)
            py = int(nose_pos[1] * view_height)
            cv2.circle(display_frame, (px, py), 6, (40, 230, 90), 2)
            cv2.circle(display_frame, (px, py), 3, (30, 200, 60), -1)

        surface = pygame.image.frombuffer(display_frame.tobytes(), (view_width, view_height), "RGB")
        cam_x = self.play_width + 10
        cam_y = self.height - view_height - 20
        pygame.draw.rect(self.screen, (32, 36, 48), (cam_x - 4, cam_y - 4, view_width + 8, view_height + 8), 2)
        label = self.font.render("Camera", True, (210, 210, 220))
        self.screen.blit(label, (cam_x, cam_y - 28))
        self.screen.blit(surface, (cam_x, cam_y))

    def draw_items(self) -> None:
        for item in self.items:
            item.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)
        for text in self.texts:
            text.draw(self.screen, self.font)

    def draw_player(self) -> None:
        pygame.draw.rect(self.screen, (90, 180, 255), self.player_rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), self.player_rect, 2, border_radius=8)

    def draw_flash(self) -> None:
        if self.flash_timer > 0:
            overlay = pygame.Surface((self.play_width, self.height))
            overlay.set_alpha(int(180 * (self.flash_timer / 0.3)))
            overlay.fill((150, 0, 0))
            self.screen.blit(overlay, (0, 0))

    def draw_game_over(self) -> None:
        if not self.game_over:
            return
        overlay = pygame.Surface((self.play_width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        text = self.big_font.render("Thua rồi!", True, (255, 200, 200))
        score_text = self.font.render(f"Điểm: {self.score}", True, (230, 230, 240))
        hint = self.font.render("Nhấn R để chơi lại hoặc ESC để thoát", True, (230, 230, 240))
        self.screen.blit(text, (self.play_width // 2 - text.get_width() // 2, 200))
        self.screen.blit(score_text, (self.play_width // 2 - score_text.get_width() // 2, 260))
        self.screen.blit(hint, (self.play_width // 2 - hint.get_width() // 2, 310))

    def run(self) -> None:
        while True:
            dt = self.clock.tick(60) / 1000.0
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
            nose = self.tracker.detect(frame)
            self.update_player(nose)

            elapsed = time.time() - self.start_time
            level = int(elapsed // 15)
            speed_factor = 1.0 + level * 0.25
            spawn_interval = max(0.55, 1.2 - level * 0.1)
            self.last_spawn += dt
            if not self.game_over and self.last_spawn >= spawn_interval:
                self.spawn_item(speed_factor)
                self.last_spawn = 0.0

            if not self.game_over:
                for item in self.items[:]:
                    item.update(dt)
                    if item.y > self.height + 40:
                        self.items.remove(item)
                self.handle_collisions()

            self.update_effects(dt)
            if self.flash_timer > 0:
                self.flash_timer -= dt

            self.draw_background()
            self.draw_items()
            self.draw_player()
            self.draw_flash()
            self.draw_scoreboard()
            self.draw_camera_view(nose)
            self.draw_game_over()
            pygame.display.flip()


if __name__ == "__main__":
    game = MineCartGame()
    game.run()

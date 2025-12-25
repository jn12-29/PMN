"""
迷宫导航环境的工具函数和常量定义
"""

from enum import Enum
import numpy as np


OBS_DICT = {
    "OBS_START_POS": 0,
    "OBS_TARGET_POS": 1,
    "OBS_WALL": 2,
    "OBS_ROAD": 3,
}


class Direction(Enum):
    """移动方向枚举"""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DEFAULT_MAZE_LAYOUT = """
*************
*   *       *
*   *   *****
*           *
**  **  **  *
*           *
*   *****   *
*   *****   *
*   *****   *
*   *****   *
*************
"""[
    1:
]


def find_positions(entity_layer, rows, cols):
    """从迷宫中查找起点和目标位置"""
    spawn_positions = []
    object_positions = []

    for row in range(rows):
        for col in range(cols):
            cell = entity_layer[row][col]
            if cell == "P":
                spawn_positions.append((row, col))
            elif cell == "G":
                object_positions.append((row, col))

    return spawn_positions, object_positions


def encode_cell(cell):
    """
    编码格子类型

    Args:
        cell: 格子字符

    Returns:
        int: 编码后的数字
    """
    if cell == "*":
        return OBS_DICT["OBS_WALL"]
    elif cell == " ":
        return OBS_DICT["OBS_ROAD"]
    elif cell == "P":
        return OBS_DICT["OBS_START_POS"]
    elif cell == "G":
        return OBS_DICT["OBS_TARGET_POS"]
    else:
        raise ValueError(f"未知格子类型: {cell}")


def render_frame(
    entity_layer,
    current_pos,
    render_mode=None,
    window=None,
    clock=None,
    render_fps=1,
):
    rows = len(entity_layer)
    cols = len(entity_layer[0])
    """渲染一帧"""
    import pygame

    if window is None:
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Maze Navigation")

    if clock is None:
        clock = pygame.time.Clock()

    # 计算格子大小
    cell_size = min(400 // cols, 400 // rows)

    # 创建画布
    canvas = pygame.Surface((cell_size * cols, cell_size * rows))
    canvas.fill((255, 255, 255))

    # 绘制迷宫
    for row in range(rows):
        for col in range(cols):
            cell = entity_layer[row][col]
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)

            if cell == "*":
                # 墙壁 - 黑色
                pygame.draw.rect(canvas, (0, 0, 0), rect)
            elif cell == " ":
                # 通道 - 白色
                pygame.draw.rect(canvas, (255, 255, 255), rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)
            elif cell == "P":
                # 起点 - 蓝色
                pygame.draw.rect(canvas, (0, 0, 255), rect)
            elif cell == "G":
                # 目标 - 绿色
                pygame.draw.rect(canvas, (0, 255, 0), rect)

    # 绘制当前位置 - 红色
    current_row, current_col = current_pos
    current_rect = pygame.Rect(
        current_col * cell_size + cell_size // 4,
        current_row * cell_size + cell_size // 4,
        cell_size // 2,
        cell_size // 2,
    )
    pygame.draw.rect(canvas, (255, 0, 0), current_rect)

    if render_mode == "human":
        # 将画布居中显示在窗口中
        window.fill((50, 50, 50))

        # 计算居中位置
        canvas_width = cell_size * cols
        canvas_height = cell_size * rows
        x_offset = (400 - canvas_width) // 2
        y_offset = (400 - canvas_height) // 2

        window.blit(canvas, (x_offset, y_offset))
        pygame.event.pump()
        pygame.display.update()
        clock.tick(render_fps)
        return window, clock
    else:  # rgb_array
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

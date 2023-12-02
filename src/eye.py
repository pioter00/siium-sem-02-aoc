from enum import Enum
from turtle import position


class Direction(Enum):
    MIDDLE = 0,
    UP = 1,
    RIGHT = 2,
    DOWN = 3,
    LEFT = 4


class Eye:
    def __init__(self, frame: list, middle_block=(20, 20)) -> None:
        self.frame = frame
        self.middle_block = middle_block

    def get_center_of_frame(self) -> tuple:
        return len(self.frame) // 2, len(self.frame[0]) // 2

    def set_starting_position(self, position: tuple):
        self.starting_position = position

    def set_eye_position(self, position: tuple):
        self.eye_position = position

    def get_direction(self) -> Direction:
        x_current, y_current = self.eye_position
        x_start, y_start = self.starting_position

        # print(self.starting_position, self.eye_position)
        if (x_current + self.middle_block[0] > x_start > x_current - self.middle_block[0] and
                y_current + self.middle_block[1] > y_start > y_current - self.middle_block[1]):
            return Direction.MIDDLE

        elif y_start >= y_current + self.middle_block[1]:
            print(f'Looking Up')
            return Direction.UP

        elif y_start <= y_current - self.middle_block[1]:
            print(f'Looking Down')
            return Direction.DOWN

        elif x_start >= x_current + self.middle_block[0]:
            print(f'Looking right')
            return Direction.RIGHT

        elif x_start <= x_current - self.middle_block[0]:
            print(f'Looking left')
            return Direction.LEFT

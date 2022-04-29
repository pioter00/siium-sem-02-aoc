from queue import Queue
from threading import Thread
from eye import Direction
import pyautogui

class Scroller(Thread):
    def __init__(self, data_queue: Queue, fps = 25, scroll_ticks = 10) -> None:
        self.data_queue = data_queue
        self.fps = fps
        self.scroll_ticks = scroll_ticks
        return super().__init__()

    def run(self) -> None:
        self.counter = 0
        self.last_directions = []
        while True:
            directions = self.data_queue.get()
            if directions == None:
                break
            self.manage_scrolling(directions)
            self.last_directions = directions

    def manage_scrolling(self, directions):
        if len(directions) != 2 or not self.last_directions:
            return

        if directions[0] == Direction.MIDDLE and directions[1] == Direction.MIDDLE:
            return

        if directions == self.last_directions and directions[0] == directions[1]:
            self.counter += 1
        else:
            self.counter = 0
        
        if self.counter >= self.fps:
            self.scroll(directions[0])
            self.counter = 0

    def scroll(self, direction: Direction):
        if direction == Direction.UP:
            pyautogui.scroll(self.scroll_ticks)
            print('Scrolling UP')

        elif direction == Direction.DOWN:
            pyautogui.scroll(-self.scroll_ticks)
            print('Scrolling DOWN')
            
        elif direction == Direction.RIGHT:
            pyautogui.hscroll(self.scroll_ticks)
            print('Scrolling RIGHT')

        elif direction == Direction.LEFT:
            pyautogui.hscroll(-self.scroll_ticks)
            print('Scrolling LEFT')
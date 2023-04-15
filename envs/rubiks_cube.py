from typing import Literal, Tuple

import numpy as np
from environment import Environment


class RubiksCube(Environment):
    """
    Class for Rubik's cube environment

    """

    def __init__(self, size: Tuple[int, int] = (3, 3), seed=None) -> None:
        self.height = size[0]
        self.width = size[1]
        self.color_to_int = {"g": 1, "r": 2, "b": 3, "o": 4, "w": 5, "y": 6}
        self.int_to_color = {v: k for k, v in self.color_to_int.items()}

        self.left = self._init_side("g")
        self.center = self._init_side("r")
        self.right = self._init_side("b")
        self.rightright = self._init_side("o")
        self.top = self._init_side("w")
        self.bottom = self._init_side("y")

        print("Initial cube before randomizing:")
        self.render()

    def step(self, action):
        pass

    def reset(self, n_actions):
        """Samples #n_actions random actions to be applied to the solved cube"""
        pass

    def render(self) -> None:
        cube_render = np.zeros(
            (3 * self.height + 2, 4 * self.width + 4), dtype="S1"
        )

        # EOF
        cube_render[:, -1] = "\n"

        # Build the render (matrix-indexed)
        # 1st block (rows)
        cube_render[: self.height, : self.width] = " "
        cube_render[: self.height, self.width] = "|"
        cube_render[
            : self.height, self.width + 1 : 2 * self.width + 1
        ] = self._face_to_col("top")
        cube_render[: self.height, 2 * self.width + 1] = "|"
        cube_render[: self.height, 2 * self.width + 2 : -1] = " "

        # horizontal separator
        cube_render[self.height, :-1] = "-"

        # 2nd block (rows)
        cube_render[
            self.height + 1 : 2 * self.height + 1, : self.width
        ] = self._face_to_col("left")
        cube_render[self.height + 1 : 2 * self.height + 1, self.width] = "|"

        cube_render[
            self.height + 1 : 2 * self.height + 1,
            self.width + 1 : 2 * self.width + 1,
        ] = self._face_to_col("center")
        cube_render[
            self.height + 1 : 2 * self.height + 1, 2 * self.width + 1
        ] = "|"

        cube_render[
            self.height + 1 : 2 * self.height + 1,
            2 * self.width + 2 : 3 * self.width + 2,
        ] = self._face_to_col("right")
        cube_render[
            self.height + 1 : 2 * self.height + 1, 3 * self.width + 2
        ] = "|"

        cube_render[
            self.height + 1 : 2 * self.height + 1, 3 * self.width + 3 : -1
        ] = self._face_to_col("rightright")

        # horizontal separator
        cube_render[2 * self.height + 1, :-1] = "-"

        # 3rd block (rows)
        cube_render[2 * self.height + 2 :, : self.width] = " "
        cube_render[2 * self.height + 2 :, self.width] = "|"

        cube_render[
            2 * self.height + 2 :, self.width + 1 : 2 * self.width + 1
        ] = self._face_to_col("bottom")
        cube_render[2 * self.height + 2 :, 2 * self.width + 1] = "|"

        cube_render[2 * self.height + 2 :, 2 * self.width + 2 : -1] = " "

        # render
        cube_render = [b.decode("UTF-8") for row in cube_render for b in row]
        print("".join(cube_render))

    def _init_side(self, c: str) -> np.array:
        content = [
            [self.color_to_int[c] for _ in range(self.width)]
            for _ in range(self.height)
        ]

        return np.array(content)

    def _face_to_col(
        self,
        face: Literal[
            "left", "center", "right", "rightright", "top", "bottom"
        ],
    ) -> np.array:
        char_face = np.zeros((self.height, self.width), dtype="S1")
        for i in range(self.height):
            for j in range(self.width):
                char_face[i, j] = self.int_to_color[getattr(self, face)[i][j]]

        return char_face


if __name__ == "__main__":
    rubics_cube = RubiksCube((4, 4))

    print("All tests have passed successfully!")

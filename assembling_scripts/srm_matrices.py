import numpy


def get_kernels():
    SRM = numpy.array(
        [
            [  # 1
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 2
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 3
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 4
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 5
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 6
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 7
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 8
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 9
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 10
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -2, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 11
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -2, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 12
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -2, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 13
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -3, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 14
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [  # 15
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            [  # 16
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, -3, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            [  # 17
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, -3, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 18
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 19
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 20
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, -3, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 21
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 22
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 23
                [0, 0, 0, 0, 0],
                [0, -1, 2, 0, 0],
                [0, 2, -4, 0, 0],
                [0, -1, 2, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 24
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 25
                [0, 0, 0, 0, 0],
                [0, 0, 2, -1, 0],
                [0, 0, -4, 2, 0],
                [0, 0, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 26
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            [  # 27
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [  # 28
                [0, 0, -2, 2, -1],
                [0, 0, 8, -6, 2],
                [0, 0, -12, 8, -2],
                [0, 0, 8, -6, 2],
                [0, 0, -2, 2, -1],
            ],
            [  # 29
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            [  # 30
                [-1, 2, -2, 0, 0],
                [2, -6, 8, 0, 0],
                [-2, 8, -12, 0, 0],
                [2, -6, 8, 0, 0],
                [-1, 2, -2, 0, 0],
            ]
        ]
    )
    return SRM

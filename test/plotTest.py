import matplotlib.pyplot as plt

a=[15434, 10867, 5343, 2345, 1978, 1543
]
b=[7864, 2343, 1234, 534, 464, 234
]
c=[4019, 331, 190, 167, 156, 123
]
d=[10765, 4987, 3345, 2367, 1455, 897, 608, 398, 298, 267, 257]
e=[14732.5, 9037, 7700.5, 6967, 5755.5, 4496.5, 4293, 3427, 2182.5, 1472, 1082.5]
f=[14987, 3657, 2879, 1056, 709, 506, 305, 287, 276, 256 ,235]

plt.plot(a, marker="o",markersize=10, linewidth=3, linestyle="-", color="blue", label=r'DQN with $\varepsilon$-greedy $\varepsilon$=0.1')
plt.plot(b, marker="p", markersize=10, linewidth=3, linestyle="-", color="red",
         label=r"DQN with intrinsic motivation $r_b$=-1")
plt.plot(c, marker="D",markersize=10, linewidth=3, linestyle="-", color="cyan", label=r"DQN-TEB")

plt.legend()
font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
plt.title("Learning curves in Modified Mountain Car", font2)
plt.xlabel(r"Episode$\times$5", font2)
plt.ylabel("Total steps per episode", font2)
plt.show()
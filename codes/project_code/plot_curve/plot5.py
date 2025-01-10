import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter

plt.style.use(['science','no-latex'])


file_path = ""  
data = pd.read_csv(file_path)

episodes = data['episode'][:100]
algorithms = ["DQN", "PPO"]

for algorithm in algorithms:
    reward_mean = data[f'{algorithm} - reward'][:100]
    reward_mean_smooth = savgol_filter(reward_mean, window_length=3, polyorder=2)
    plt.plot(episodes, reward_mean_smooth, label=f'{algorithm}')

plt.title("Training Episode Returns Across Episodes")
plt.xlabel("Episodes")
plt.ylabel("Episdoe Returns")
plt.legend()
plt.tight_layout()

plt.savefig("output/training_rewards5.pdf", format="pdf", bbox_inches="tight")
plt.savefig("output/training_rewards5.png", dpi=300, bbox_inches='tight')
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter

plt.style.use(['science','no-latex'])


file_path = ""  
data = pd.read_csv(file_path)

episodes = data['Step']
algorithms = ["DQN", "PPO"]

for algorithm in algorithms:
    reward_mean = data[f'{algorithm} - agent0 reward']
    reward_mean_smooth = reward_mean
    plt.plot(episodes, reward_mean_smooth, label=f'{algorithm}')

plt.title("ManeuverAgent Reward Across Steps")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.tight_layout()

plt.savefig("output/maneuverrewards.pdf", format="pdf", bbox_inches="tight")
plt.savefig("output/maneuverrewards.png", dpi=300, bbox_inches='tight')
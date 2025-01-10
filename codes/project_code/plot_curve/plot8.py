import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter

plt.style.use(['science','no-latex'])

for agent in ["agent0", "agent1"]:
    file_path = "" if agent == "agent0" else ""
    data = pd.read_csv(file_path)
    episodes = data['episode']
    reward_mean = data[f'distinctive-lake-1 - {agent} reward']
    reward_mean_smooth = savgol_filter(reward_mean, window_length=5, polyorder=2)
    plt.plot(episodes, reward_mean_smooth, label=f'{agent}')

plt.title("Training Episode Returns Across Episodes")
plt.xlabel("Episodes")
plt.ylabel("Episdoe Returns")
plt.legend()
plt.tight_layout()

plt.savefig("output/training_rewards7.pdf", format="pdf", bbox_inches="tight")
plt.savefig('output/training_rewards7.png', dpi=300, bbox_inches='tight')
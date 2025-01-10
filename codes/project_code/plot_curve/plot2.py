import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter

plt.style.use(['science','no-latex'])


file_path = ""  
data = pd.read_csv(file_path)

episodes = data['episode']
hidden_dim = ['128', '64', '32']

for hd in hidden_dim:
    reward_mean = data[f'hidden_dim={hd} - reward']
    reward_mean_smooth = savgol_filter(reward_mean, window_length=5, polyorder=2)
    plt.plot(episodes, reward_mean_smooth, label=f'hidden dim={hd}')

plt.title("Training Episode Returns Across Episodes")
plt.xlabel("Episodes")
plt.ylabel("Episdoe Returns")
plt.legend()
plt.tight_layout()

plt.savefig("output/training_rewards2.pdf", format="pdf", bbox_inches="tight")
plt.savefig("output/training_rewards2.png", dpi=300, bbox_inches='tight')
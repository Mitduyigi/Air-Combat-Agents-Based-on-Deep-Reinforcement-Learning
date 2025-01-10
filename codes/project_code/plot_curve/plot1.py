import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter

plt.style.use(['science','no-latex'])


file_path = ""  
data = pd.read_csv(file_path)

episodes = data['episode']
learning_rates = ['5e-5', '1e-4', '5e-4', '1e-3']

for lr in learning_rates:
    reward_mean = data[f'learning_rate={lr} - reward']
    reward_mean_smooth = savgol_filter(reward_mean, window_length=5, polyorder=2)
    plt.plot(episodes, reward_mean_smooth, label=f'learning rate={lr}')

plt.title("Training Episode Returns Across Episodes")
plt.xlabel("Episodes")
plt.ylabel("Episdoe Returns")
plt.legend()
plt.tight_layout()

plt.savefig("output/training_rewards1.pdf", format="pdf", bbox_inches="tight")
plt.savefig("output/training_rewards1.png", dpi=300, bbox_inches='tight')
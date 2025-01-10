import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from scipy.signal import savgol_filter

plt.style.use(['science','no-latex'])


file_path = ""  
data = pd.read_csv(file_path)

episodes = data['episode']
improvements = ['Vanilla', 'Double', 'Prioritized', 'Dueling', 'Double+Prioritized+Dueling']

for improvement in improvements:
    reward_mean = data[f'{improvement} - reward']
    reward_mean_smooth = savgol_filter(reward_mean, window_length=5, polyorder=2)
    if improvement == 'Double+Prioritized+Dueling':
        plt.plot(episodes, reward_mean_smooth, label=f'3 improvements')
    else:
        plt.plot(episodes, reward_mean_smooth, label=f'{improvement}')

plt.title("Training Episode Returns Across Episodes")
plt.xlabel("Episodes")
plt.ylabel("Episdoe Returns")
plt.legend()
plt.tight_layout()

plt.savefig("output/training_rewards3.pdf", format="pdf", bbox_inches="tight")
plt.savefig("output/training_rewards3.png", dpi=300, bbox_inches='tight')
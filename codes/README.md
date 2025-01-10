## Code Explanation
### project_code
#### SingleConteol Environment
In the `algorithm` folder, `dqn.py` and `scripts.py` are the basic code for the DQN algorithm and training. The files under the `algorithm/joint` folder provide the basic code for the DQN algorithm and training, with the MultiDiscrete action space fully expanded to Discrete acion space.

The `experiment` folder contains the training and evaluating codes for tuning hyperparameters and exploring different DQN improvements.

The `agent` folder implements the Maneuver & Pursue baseline and its visualization using agents trained in the SingleControl environment.

The `render` folder contains codes for simply rendering the environment.

#### SingleCombat Environment
The `single_combat` folder contains the implementation of the Maneuver & Pursue task in the SingleCombat environment with the NoWeapon task and the self-play training setting.

#### Plotting Curves
The `plot_curve` folder contains the codes for generating the plots shown in this project's technical report and presentation slides.
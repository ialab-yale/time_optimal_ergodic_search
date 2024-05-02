## Setup
Open two terminal windows and source ROS Noetic in both of them. Then, make sure to also run ```source develop/setup.zsh``` from the project root directory in both windows.

## Running Experiment
In one of the terminals, run ```roslaunch time_optimal_ergodic_search multiscale_viz.launch``` from the project root directory and rviz should open up. From here, on the dropdown in the top right hand corner you should be able to select what view you want. Orbit and TopDownOrtho are two good ones we have been using.

In the second terminal, run ```python3 render_traj_stills.py``` and now the figure should show up in rviz.
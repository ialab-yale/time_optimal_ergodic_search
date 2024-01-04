# Minimum Distance Ergodic Search

This is our subfolder dedicated to the minimum distance ergodic search optimization problem. While also used for the Time Optimal Ergodic Search Paper[^1], this is also a final project for Professor Daniel Rakita's CPSC 485 Applied Planning and Optimization class.

The problem setup and solver are in `build_solver.py`, and the main file to run to perform the optimization is `render_bias_search.py`.

## Running the Code
To run the code you will need to source ROS Noetic and also run `source devel/setup.zsh` in from the root ros directory. Then you will need `roscore` running in a terminal as well as `rosrun rviz rviz` and `python3 drone_viz.py` all before running `python3 render_bias_search.py`

[^1]:Dong, D., Berger, H., & Abraham, I. (2023). Time Optimal Ergodic Search (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2305.11643
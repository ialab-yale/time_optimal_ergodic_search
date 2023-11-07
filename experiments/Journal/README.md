# Instructions

If it is your first time downloading the repository, make sure that you have the [drone_env_viz](https://github.com/i-abr/drone_vis) repository cloned in the src folder of the workspace and make sure it is named "drone_env_viz".

For every new terminal window you open, run the following commands:
```
source devel/setup.zsh  # run this in the workspace folder
```
```
source /opt/ros/noetic/setup.zsh    # run this anywhere
```


## complex_clutter
Start three terminals and make sure to run the commands above in each. In the first one, run:
```
roscore
```
in the second one, run:
```
rosrun rviz rviz
```
when the rviz window opens, go to File/Open Config and open src/time_optimal_ergodic_search/rviz/cluttered_env.rviz. Then, in the third, while in the complex_clutter folder, run:
```
python3 render_traj_stills.py
```

## bias_search
Start four terminals and make sure to run the commands above in each. In the first one, run:
```
roscore
```
in the second one, run:
```
rosrun rviz rviz
```
when the rviz window opens, go to File/Open Config and open src/time_optimal_ergodic_search/rviz/bias_search.rviz. Then, in the third, while in the complex_clutter folder, run:
```
python3 drone_viz.py
```
and in the last one, while in the bias_search folder, run:
```
python3 render_bias_search.py
```
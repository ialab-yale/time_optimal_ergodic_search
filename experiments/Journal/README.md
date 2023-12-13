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
If you are running the simulation, first make sure you run tor_convert_obs2numpy.py and box_convert_obs2numpy.py if you want to change the obstacles.

Next, start three terminals and make sure to run the commands above in each. In the first one, run:
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
or
```
python3 render_traj.py
```
but the second is the more updated file but the first one can easily be adapted for the desired setup.


In order to run this on the drone, first make sure that you source ROS2. Then, enter the drone folder and simultaneously run the robot.py file with the correct ID for the quadcopter and the trajectory.py file with the correct trajectory for the given environment. These trajectories can be generated using render_traj.py or render_traj_stills.py.



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
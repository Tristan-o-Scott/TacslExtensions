# TacslExtensions
Custom extensions made for Isaac Sim.
## MPlib Example
Clone repo into desired location. Once it has been cloned, create symlink into the `exts/` in your Isaac Sim installation directory.
```
ln -s /path/to/MPlib_extension/exts ~/isaacsim-4.5.0/exts/MPlib_extension
```

Once you've done that, run Isaac Sim GUI from terminal.

When Isaac Sim is loaded, go to:
Window -> Extensions ... from here type `mplib` into the search bar. Enable the following extension:
```
FRANKA PICK-AND-PLACE TASK  
```

Now, click on:
Window -> Examples -> Robotics Examples

There should be a tab now in Robotics Examples at the very bottom titled `CUSTOM EXAMPLES`. Click on the file and click `LOAD` to load the scene. You should see a Franka Panda arm with 3 cubes.

To run the simulation, press the play button to the left before clicking on `RUN` in the Task Control window (bottom of the screen).
# TacslExtensions
Custom extensions made for Isaac Sim.
# MPlib Example
Clone repo into desired location. Once it has been cloned, create symlink into the `exts/` in your Isaac Sim installation directory.
```
ln -s /path/to/MPlib_extension/exts ~/isaacsim-4.5.0/exts/MPlib_extension
```
Or you can copy contents from `TacslExtensions/exts/MPlib_extension` into `isaacsim-4.5.0/exts/MPlib_extension` directly.

## Install MPlib
To install MPlib so that Isaac Sim can recognize it, do the following at the top of your Isaac Sim directory.
```bash
./python.sh -m pip install mplib
```
if you are using Linux.

## Run the script
Once you've done that, run Isaac Sim GUI from terminal.

When Isaac Sim is loaded, go to:
Window -> Extensions ... from here type `mplib` into the search bar. Enable the following extension:
```
FRANKA PICK-AND-PLACE TASK  
```
To enable this extension for all times you open Isaac Sim, click on the checkmark for "Autoload".

Now, click on:
Window -> Examples -> Robotics Examples

There should be a tab now in Robotics Examples at the very bottom titled `CUSTOM EXAMPLES`. Click on the file and click `LOAD` to load the scene. You should see a Franka Panda arm with 3 cubes.

To test your installation of MPlib, try running `RUN TEST`. If you see the Franka arm moving, then MPlib is working correctly.

To run the simulation, press the play button to the left before clicking on `RUN` in the Task Control window (bottom of the screen).

# Block Throw Example
The target zone for where the blocks are to be thrown to can be changed according to what is needed for testing.

The blocks may not land within the target zone due to them slipping while thrown, the behavior of the physics, or the blocks not settling when being grasped.

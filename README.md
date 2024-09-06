# Interactive fractal explorer in matplotlib
This fractal explorer uses key inputs to move around a fractal. The fractal defined by convergence of uniformly random points in the complex plane under some repeated map `F`

## Controls
- Arrow keys: pan in the complex plane, i.e. <kbd>up</kbd> = moves the center <math>+k*i</math> units, where <math>k</math> depends on the current zoom level.
- <kbd>i</kbd> - zoom in
- <kbd>k</kbd> - zoom out
- <kbd>+</kbd> - rerender with more points ("more detail")
- <kbd>-</kbd> - rerender with less points ("less detail")
The above two are useful for quicker exploration of complex fractals and allow the user to get the target viewport faster
- <kbd>r</kbd> - rerender without any changes (generates new random points)
- <kbd>⌘ Command</kbd> + <kbd>r</kbd> - reset everything
- <kbd>q</kbd> - quit

## TODO
- Code cleanup
- Make it look nicer lol
- Remove unnecessary logs
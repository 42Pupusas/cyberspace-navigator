# Cyberspace Navigator

An exploratory implementation of [Cyberspace Protocol](https://github.com/arkin0x/cyberspace), 
using the [Bevy](https://bevyengine.org/) [Rust](https://www.rust-lang.org/) engine.
Pulls notes from a few public relays, and maps them to a 3D grid. Unique pubkeys, or `Avatars`,
are displayed as bright spheres. Notes and reactions are mapped and scaled to have their origin on their Avatar, and 
displayed as smaller darker spheres.

## Demo

You can find a simple demo of the viewer on [Rust](https://www.rust-lang.org/)

## Controls

Move around cyberspace using your mouse and keys.

### Camera

Hold the right mouse button to swivel the camera around.

### Movement

Use the `WASD` keys to move forwards, backwards and sideways. You can hold the spacebar to accelerate.

### Interactions

Click on Notes to read their content on the terminal. Clicking on Avatars will despawn them.
These are experimental features and will likely change.



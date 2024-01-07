# Cyberspace Navigator

An exploratory implementation of [Cyberspace Protocol](https://github.com/arkin0x/cyberspace), using [Bevy](https://bevyengine.org/) as an engine.
Pulls notes from a few public relays, and maps them to a 3D grid. Unique pubnkeys, or `Avatars`,
are displayed as blue spheres. Notes are mapped and scaled to have their origin on their Avatar, and 
displayed as smaller green spheres.

## Demo

You must have [Rust](https://www.rust-lang.org/) installed on your machine to run the demo.

Clone the repo and run a precompiled version of the app using the following command:

```
cargo run
```

PLEASE NOTE: It takes quite a while the first time you run it as bevy precompiles most of its code before running the app.

## Controls

Move around cyberspace using your mouse and keys.

### Camera

Hold the right mouse button to swivel the camera around.

### Movement

Use the `WASD` keys to move forwards, backwards and sideways. You can hold the spacebar to accelerate.

### Interactions

Click on Notes to read their content on the terminal. Clicking on Avatars will despawn them.
These are experimental features and will likely change.

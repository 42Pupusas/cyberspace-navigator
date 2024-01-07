use bevy::{input::mouse::MouseMotion, prelude::*, window::PresentMode};

use bevy_mod_picking::{
    prelude::{Click, ListenerInput, On, Out, Over, Pointer},
    DefaultPickingPlugins,
};

use primitive_types::U256;
use serde_json::json;

use cryptoxide::digest::Digest;
use cryptoxide::sha2::Sha256;
use std::collections::{HashMap, HashSet};

use nostro2::{
    notes::SignedNote,
    relays::{NostrRelay, RelayEvents},
};

use bevy_async_task::AsyncTaskPool;
use crossbeam_channel::{bounded, Receiver};

const WINDOW_WIDTH: f32 = 1280.0;
const WINDOW_HEIGHT: f32 = 720.0;

// Main function loop, runs once on startup 
fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Cyberspace Navigator".into(),
                    resolution: (WINDOW_WIDTH, WINDOW_HEIGHT).into(),
                    present_mode: PresentMode::AutoVsync,
                    prevent_default_event_handling: true,
                    resizable: true,
                    fit_canvas_to_parent: true,
                    ..default()
                }),
                ..default()
            }),
            // Gives us clickin' and pickin' capabilities
            DefaultPickingPlugins,
        ))
        .insert_resource(ClearColor(Color::BLACK)) // Set background color to black
        .insert_resource(RelayCounter(0)) // Counter for the number of relays we've connected to
        .insert_resource(UniqueNotes::default()) // Set of unique notes we've received
        .insert_resource(UniqueAvatars::default()) // Set of unique avatars we've received
        // Events work as a way to pass data between systems
        .add_event::<NostrBevyEvents>()
        .add_event::<NoteBevyEvents>()
        .add_event::<DespawnAvatar>()
        // Systems are functions that run every frame
        .add_systems(Startup, (setup, cyberspace_websocket))
        .add_systems(
            Update,
            (
                camera_movement_system,
                cyberspace_middleware,
                rendering_entity_system,
                make_notes_children_of_avatar,
                despawn_avatar_system.run_if(on_event::<DespawnAvatar>()),
                rotate_cyberspace_entities,
            ),
        )
        .run();
}

// We start the camera at a fixed position 
// far away to get a good view of the scene
fn setup(mut commands: Commands) {
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(-2000.0, 2000.0, 2000.0)
                .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            ..default()
        })
        .insert(CameraState::default());
}

#[derive(Component, Resource, Copy, Clone)]
struct CameraState {
    pub is_active: bool,
    pub speed: f32,
    pub acceleration: f32,
    pub acceleration_time: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        CameraState {
            is_active: true,
            speed: 124.0,
            acceleration: 9.8,
            acceleration_time: 0.0,
        }
    }
}

fn camera_movement_system(
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
    mouse_input: Res<Input<MouseButton>>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut camera_state: Query<&mut CameraState, With<CameraState>>,
    mut query: Query<&mut Transform, With<CameraState>>,
) {
    let sensitivity = 0.42;
    let mut camera_transform = query.single_mut();
    let mut camera_state = camera_state.single_mut();

    if mouse_input.pressed(MouseButton::Right) {
        if camera_state.is_active {
            camera_state.is_active = false;
            mouse_motion_events.clear(); // Clear the accumulated events
            return; // Skip the rest of the logic for this frame
        }

        let delta: Vec2 = mouse_motion_events
            .read()
            .fold(Vec2::ZERO, |acc, motion| acc + motion.delta);

        // Update camera rotation based on mouse movement
        let yaw = Quat::from_rotation_y(-delta.x * sensitivity * time.delta_seconds());
        let pitch = Quat::from_rotation_x(-delta.y * sensitivity * time.delta_seconds());
        camera_transform.rotation = yaw * camera_transform.rotation * pitch;
    } else {
        camera_state.is_active = true; // Reset the flag when the button is not pressed
    }

    // accelerate when holding spacebar
    if keyboard_input.pressed(KeyCode::Space) {
        camera_state.acceleration_time += time.delta_seconds();
        camera_state.speed =
            camera_state.speed + camera_state.acceleration * camera_state.acceleration_time;
    } else {
        // TODO: this is a hacky way to reset the speed
        // should implement slowly decreasing the speed
        camera_state.speed = 124.0;
        camera_state.acceleration_time = 0.0;
    }

    // Forward movement
    if keyboard_input.pressed(KeyCode::W) {
        let forward = camera_transform.rotation.mul_vec3(Vec3::Z).normalize();
        camera_transform.translation -= forward * camera_state.speed * time.delta_seconds();
    }

    // Backward camera_movement
    if keyboard_input.pressed(KeyCode::S) {
        let forward = camera_transform.rotation.mul_vec3(Vec3::Z).normalize();
        camera_transform.translation += forward * camera_state.speed * time.delta_seconds();
    }

    // Left camera_movement
    if keyboard_input.pressed(KeyCode::A) {
        let forward = camera_transform.rotation.mul_vec3(Vec3::X).normalize();
        camera_transform.translation -= forward * camera_state.speed * time.delta_seconds();
    }

    // Right camera_movement
    if keyboard_input.pressed(KeyCode::D) {
        let forward = camera_transform.rotation.mul_vec3(Vec3::X).normalize();
        camera_transform.translation += forward * camera_state.speed * time.delta_seconds();
    }
}

// Will rotate Avatars every frame
// Because Notes are children of Avatrs, they will rotate as well 
fn rotate_cyberspace_entities(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<CyberspaceMarker>>,
) {
    for mut transform in query.iter_mut() {
        transform.rotate(Quat::from_rotation_y(time.delta_seconds() * 0.042));
        transform.rotate(Quat::from_rotation_x(time.delta_seconds() * 0.042));
    }
}

#[derive(Event, Resource)]
struct NostrBevyEvents(String);

#[derive(Event, Resource)]
struct NoteBevyEvents(CyberspaceNote);

// We use these receivers to hold the data that we receive from the websocket
#[derive(Resource, Deref)]
struct NostrReceiver(Receiver<String>);

#[derive(Resource, Deref)]
struct NoteReceiver(Receiver<CyberspaceNote>);

const RELAY_LIST: [&str; 5] = [
    "wss://relay.arrakis.lat",
//    "wss://nostr.wine",
    "wss://njump.me",
    "wss://nostr-pub.wellorder.net",
//    "wss://purplerelay.com",
//    "wss://relay.n057r.club",
    "wss://nostr.maximacitadel.org",
//    "wss://relay.geekiam.services",
//    "wss://nostr.jcloud.es",
    "wss://relay.hodl.ar",
];

fn cyberspace_websocket(mut commands: Commands, mut task_pool: AsyncTaskPool<()>) {
    let (relay_tx, relay_rx) = bounded::<String>(10);
    let (note_tx, note_rx) = bounded::<CyberspaceNote>(50);

    if task_pool.is_idle() {
        for relay_url in RELAY_LIST.iter() {
            let relay_tx = relay_tx.clone();
            let note_tx = note_tx.clone();
            task_pool.spawn(async move {
                if let Ok(relay) = NostrRelay::new(relay_url).await {
                    // let one_week_ago = nostro2::utils::get_unix_timestamp() - 604800;
                    let yesterday = nostro2::utils::get_unix_timestamp() - 86400;
                    let filter = json!({"kinds": [1], "since": yesterday});
                    let _ = relay.subscribe(filter).await;
                    while let Some(Ok(msg)) = relay.read_from_relay().await {
                        match msg {
                            RelayEvents::EVENT(_, _, signed_note) => {
                                let note = CyberspaceNote::new(signed_note);
                                let _ = note_tx.send(note);
                            }
                            RelayEvents::EOSE(_, _) => {
                                info!("End of stream from relay: {}", relay_url.to_string());
                                relay_tx.send(relay_url.to_string()).unwrap();
                            }
                            _ => {}
                        }
                    }
                } else {
                    info!("Failed to connect to relay: {}", relay_url.to_string());
                }
            });
        }
    }

    commands.insert_resource(NostrReceiver(relay_rx));
    commands.insert_resource(NoteReceiver(note_rx));
}

#[derive(Resource)]
struct UniqueNotes(HashSet<String>);

impl Default for UniqueNotes {
    fn default() -> Self {
        UniqueNotes(HashSet::new())
    }
}

fn cyberspace_middleware(
    nostr_receiver: Res<NostrReceiver>,
    note_receiver: Res<NoteReceiver>,
    mut nostr_writer: EventWriter<NostrBevyEvents>,
    mut note_writer: EventWriter<NoteBevyEvents>,
    mut unique_notes: ResMut<UniqueNotes>,
) {
    let note_set = unique_notes.as_mut();

    nostr_receiver.try_iter().for_each(|relay_url| {
        nostr_writer.send(NostrBevyEvents(relay_url));
    });

    note_receiver.try_iter().for_each(|note| {
        let id = note.id.clone();
        if note_set.0.contains(&id) {
            return;
        }
        note_writer.send(NoteBevyEvents(note));
        note_set.0.insert(id);
    });
}

#[derive(Resource)]
struct UniqueAvatars(HashSet<String>);

impl Default for UniqueAvatars {
    fn default() -> Self {
        UniqueAvatars(HashSet::new())
    }
}

fn rendering_entity_system(
    mut note_events: EventReader<NoteBevyEvents>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut unique_avatars: ResMut<UniqueAvatars>,
) {
    let avatar_set = unique_avatars.as_mut();

    for cyber_event in note_events.read() {
        if !avatar_set.0.contains(&cyber_event.0.pubkey) {
            avatar_set.0.insert(cyber_event.0.pubkey.clone());
            commands.spawn(AvatarBundle::new(
                cyber_event.0.pubkey.clone(),
                &mut materials,
                &mut meshes,
            ));
        }
        commands.spawn(NoteBundle::new(
            cyber_event.0.clone(),
            &mut materials,
            &mut meshes,
        ));
    }
}

#[derive(Resource)]
struct RelayCounter(usize);

fn make_notes_children_of_avatar(
    mut commands: Commands,
    mut nostr_events: EventReader<NostrBevyEvents>,
    note_query: Query<(Entity, &Kind1Note, &Transform)>,
    avatar_query: Query<(Entity, &AvatarPubkey, &Transform)>,
    mut counter: ResMut<RelayCounter>,
) {
    let counter = counter.as_mut();
    for _ in nostr_events.read() {
        counter.0 += 1;
    }

    if counter.0 == RELAY_LIST.len() {
        info!("Making notes children of avatars");
        counter.0 = 0;
        for (avatar_entity, avatar_pubkey, avatar_transform) in avatar_query.iter() {
            for (note_entity, _note, note_transform) in note_query
                .iter()
                .filter(|(_, note, _)| note.pubkey == avatar_pubkey.0)
            {
                // Calculate the relative position of the note to the avatar
                let relative_position = note_transform.translation - avatar_transform.translation;

                // Set the note's new transform to this relative position
                let relative_transform = Transform::from_translation(relative_position);

                // Attach the note as a child of the avatar
                commands.entity(avatar_entity).add_child(note_entity);

                // Apply the relative transform
                commands.entity(note_entity).insert(relative_transform);
            }
        }
    }
}

#[derive(Component)]
struct CyberspaceMarker;

#[derive(Component)]
struct AvatarPubkey(String);

#[derive(Bundle)]
struct AvatarBundle {
    pub pbr: PbrBundle,
    pub pubkey: AvatarPubkey,
    pub marker: CyberspaceMarker,
    pub pickable: (On<Pointer<Over>>, On<Pointer<Out>>, On<Pointer<Click>>),
}

impl AvatarBundle {
    pub fn new(
        pubkey: String,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) -> AvatarBundle {
        let simhash = simhash(&pubkey);
        let (x, y, z) = map_hash_to_coordinates(simhash);
        let (x_f32, y_f32, z_f32) = scale_down_coordinates_to_f32(x, y, z);

        AvatarBundle {
            pbr: PbrBundle {
                mesh: meshes.add(
                    Mesh::try_from(shape::Icosphere {
                        radius: 8.4,
                        subdivisions: 4,
                        ..Default::default()
                    })
                    .unwrap(),
                ),
                // add birght blue material
                material: materials.add(StandardMaterial {
                    base_color: Color::rgba(0.0, 0.24, 0.48, 0.0),
                    unlit: true,
                    ..default()
                }),
                transform: Transform::from_xyz(x_f32, y_f32, z_f32),
                ..Default::default()
            },
            pubkey: AvatarPubkey(pubkey),
            ..Default::default()
        }
    }
}

impl Default for AvatarBundle {
    fn default() -> Self {
        AvatarBundle {
            pbr: PbrBundle {
                ..Default::default()
            },
            pubkey: AvatarPubkey("".to_string()),
            marker: CyberspaceMarker,
            pickable: (
                On::<Pointer<Over>>::target_component_mut::<Transform>(|_over, transform| {
                    transform.scale = Vec3::splat(2.1);
                }),
                On::<Pointer<Out>>::target_component_mut::<Transform>(|_out, transform| {
                    transform.scale = Vec3::splat(1.0);
                }),
                On::<Pointer<Click>>::send_event::<DespawnAvatar>(),
            ),
        }
    }
}

#[derive(Event)]
struct DespawnAvatar(Entity);

impl From<ListenerInput<Pointer<Click>>> for DespawnAvatar {
    fn from(event: ListenerInput<Pointer<Click>>) -> Self {
        DespawnAvatar(event.target)
    }
}

fn despawn_avatar_system(
    mut commands: Commands,
    mut events: EventReader<DespawnAvatar>,
    query: Query<&AvatarPubkey>,
) {
    for event in events.read() {
        if let Ok(avatar_pubkey) = query.get(event.0) {
            info!("Avatar pubkey: {}", avatar_pubkey.0);
            commands.entity(event.0).despawn_recursive();
        }
    }
}

#[derive(Component)]
struct Kind1Note {
    pub content: String,
    pub pubkey: String,
}

#[derive(Bundle)]
struct NoteBundle {
    pub pbr: PbrBundle,
    pub note: Kind1Note,
    pub marker: CyberspaceMarker,
    pub pickable: (On<Pointer<Over>>, On<Pointer<Out>>, On<Pointer<Click>>),
}

impl Default for NoteBundle {
    fn default() -> Self {
        NoteBundle {
            pbr: PbrBundle {
                ..Default::default()
            },
            note: Kind1Note {
                content: "".to_string(),
                pubkey: "".to_string(),
            },
            marker: CyberspaceMarker,
            pickable: (
                On::<Pointer<Over>>::target_component_mut::<Transform>(|_over, transform| {
                    transform.scale = Vec3::splat(1.6);
                }),
                On::<Pointer<Out>>::target_component_mut::<Transform>(|_out, transform| {
                    transform.scale = Vec3::splat(1.0);
                }),
                On::<Pointer<Click>>::target_component_mut::<Kind1Note>(|_click, note| {
                    info!("Note says: {}", note.content);
                }),
            ),
        }
    }
}

impl NoteBundle {
    pub fn new(
        note: CyberspaceNote,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) -> NoteBundle {
        let hash = simhash(&note.content);
        let origin_hash = simhash(&note.pubkey);
        let (x, y, z) = map_hash_to_coordinates_with_offset(hash, origin_hash);

        NoteBundle {
            pbr: PbrBundle {
                mesh: meshes.add(
                    Mesh::try_from(shape::Icosphere {
                        radius: 2.1,
                        subdivisions: 4,
                        ..Default::default()
                    })
                    .unwrap(),
                ),
                material: materials.add(StandardMaterial {
                    base_color: Color::rgba(0.0, 0.48, 0.24, 0.3),
                    unlit: true,
                    ..default()
                }),
                transform: Transform::from_xyz(x, y, z),
                ..Default::default()
            },
            note: Kind1Note {
                content: note.content,
                pubkey: note.pubkey,
            },
            ..Default::default()
        }
    }
}

#[derive(Event, Clone)]
struct CyberspaceNote {
    content: String,
    _kind: u32,
    id: String,
    _signature: String,
    _created_at: u64,
    pubkey: String,
}

impl CyberspaceNote {
    pub fn new(signed_note: SignedNote) -> CyberspaceNote {
        let content = signed_note.get_content().to_string();
        let _kind = signed_note.get_kind();
        let id = signed_note.get_id().to_string();
        let _signature = signed_note.get_sig().to_string();
        let _created_at = signed_note.get_created_at();
        let pubkey = signed_note.get_pubkey().to_string();
        CyberspaceNote {
            content,
            _kind,
            id,
            _signature,
            _created_at,
            pubkey,
        }
    }
}


// CYBERSPACE METHODS
// These methods are used to generate the cyberspace coordinates for the notes and avatars 
// based on their content and public key respectively 

// Simhash is a hashing algorithm that is used to generate a 256 bit hash from a string
fn simhash(input: &str) -> U256 {
    // Initialize a vector of 256 zeros
    let mut vectors: Vec<i32> = vec![0; 256];

    // Initialize a hashmap to count the occurrences of each word
    let mut word_count = HashMap::new();

    // Split the input string into words and count their occurrences
    for word in input.split_whitespace() {
        *word_count.entry(word).or_insert(0) += 1;
    }

    // Hash each word and add/subtract from the vector
    for (word, count) in word_count {
        // Hash the word
        let hash = hash_word(word);
        for i in 0..256 {
            // Add or subtract from the vector based on the bit at index i
            // If the bit is 1, add the count, otherwise subtract
            if get_bit(hash, i) {
                vectors[i] += count;
            } else {
                vectors[i] -= count;
            }
        }
    }

    // Construct the final hash
    let mut final_hash = U256::zero();
    for i in 0..256 {
        if vectors[i] > 0 {
            final_hash = final_hash.overflowing_add(U256::one() << i).0;
        }
    }

    final_hash
}

// Get the bit at index i from a 256 bit hash
fn get_bit(hash: [u8; 32], index: usize) -> bool {
    let byte = index / 8;
    let bit = index % 8;
    hash[byte] & (1 << bit) != 0
}

// Hash a word using SHA256
fn hash_word(word: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.input_str(word);

    // Get the hash as a 32 byte array
    let mut result = [0u8; 32];
    hasher.result(&mut result);
    result
}

fn map_hash_to_coordinates(hash: U256) -> (U256, U256, U256) {
    // Split the hash into 3 parts of 85 bits each
    let mut x = U256::zero();
    let mut y = U256::zero();
    let mut z = U256::zero();

    // Map the first 85 bits to x, the next 85 to y, and the last 85 to z
    for i in 0..255 {
        match i % 3 {
            0 => x = (x << 1) | get_bit_as_u256(&hash, i),
            1 => y = (y << 1) | get_bit_as_u256(&hash, i),
            2 => z = (z << 1) | get_bit_as_u256(&hash, i),
            _ => unreachable!(),
        }
    }

    (x, y, z)
}

// U256 type comes from the primitive_types crate 
// It's a 256 bit unsigned integer
fn get_bit_as_u256(hash: &U256, index: usize) -> U256 {
    if get_bit_from_primitive(hash, index) {
        U256::one()
    } else {
        U256::zero()
    }
}

fn get_bit_from_primitive(hash: &U256, index: usize) -> bool {
    let byte_index = index / 8;
    let bit_index = index % 8;
    let byte = hash.byte(byte_index);
    (byte & (1 << bit_index)) != 0
}

// This one is used for notes, because we want to offset them from the avatar 
// we also want to scale them down to a smaller size
fn map_hash_to_coordinates_with_offset(hash: U256, origin_hash: U256) -> (f32, f32, f32) {
    // Calculate coordinates for the hash
    let (x, y, z) = map_hash_to_coordinates(hash);
    // Calculate coordinates for the origin hash
    let (origin_x, origin_y, origin_z) = map_hash_to_coordinates(origin_hash);

    let (scaled_x, scaled_y, scaled_z) = scale_down_coordinates_to_f32(x, y, z);
    let (scaled_origin_x, scaled_origin_y, scaled_origin_z) =
        scale_down_coordinates_to_f32(origin_x, origin_y, origin_z);

    let extra_scale = 42.0;

    let x_f32 = (scaled_x / extra_scale) + scaled_origin_x;
    let y_f32 = (scaled_y / extra_scale) + scaled_origin_y;
    let z_f32 = (scaled_z / extra_scale) + scaled_origin_z;

    (x_f32, y_f32, z_f32)
}

// This function scales down the coordinates to a smaller usize 
// so that we can fit them into a f32 and then scale them up to the desired
// scene 
fn scale_down_coordinates_to_f32(x: U256, y: U256, z: U256) -> (f32, f32, f32) {
    
    // Max value is 2^85
    let max_value = U256::from(1u128) << 85;

    // Extract sign and absolute value
    // The sign is the 85th bit 
    // I did this so I could get negative values as well 
    // and make the scene more interesting
    let x_sign = if x.bit(84) { -1.0 } else { 1.0 };
    let y_sign = if y.bit(84) { -1.0 } else { 1.0 };
    let z_sign = if z.bit(84) { -1.0 } else { 1.0 };

    let x_abs = x & ((U256::from(1u128) << 84) - 1);
    let y_abs = y & ((U256::from(1u128) << 84) - 1);
    let z_abs = z & ((U256::from(1u128) << 84) - 1);

    // Helper function below
    // From testing, the coordinates are always between 0 and 1.6
    // with 12 decimal points of precision
    let x_f32 = u256_to_f32(x_abs, max_value);
    let y_f32 = u256_to_f32(y_abs, max_value);
    let z_f32 = u256_to_f32(z_abs, max_value);

    // Scale the coordinates up to the desired scene size
    // This is a magic number that I picked because it looked good
    // and gave the scene a good size
    let scale = 8400.0;
    (
        x_f32 * scale * x_sign,
        y_f32 * scale * y_sign,
        z_f32 * scale * z_sign,
    )
}

fn u256_to_f32(value: U256, max_value: U256) -> f32 {
    // Convert U256 to f32 by first converting to a smaller integer (like u64) and then to f32
    // This is because U256 doesn't implement From<f32> for some reason
    //
    // We divide by max_value / u64::MAX to get a value between 0 and 1 
    // and then multiply by u64::MAX to get a value between 0 and u64::MAX
    let value_u64 = value / (max_value / U256::from(u64::MAX));
    // Once we have a value between 0 and u64::MAX, we can convert it to f32
    // by dividing by u64::MAX to get a value between 0 and 1.6
    value_u64.as_u64() as f32 / u64::MAX as f32
}

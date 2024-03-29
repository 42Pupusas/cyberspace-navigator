use bevy::{
    core_pipeline::clear_color::ClearColorConfig, input::mouse::MouseMotion, prelude::*,
    text::BreakLineOn, window::PresentMode,
};

use bevy_mod_picking::{
    prelude::{Click, ListenerInput, On, Out, Over, Pointer, PointerButton},
    DefaultPickingPlugins,
};

use nostr_sdk::{Client, Filter, Kind, RelayPoolNotification};
use primitive_types::U256;

use cryptoxide::digest::Digest;
use cryptoxide::sha2::Sha256;
use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
};

//use nostro2::{
//    notes::SignedNote,
//    relays::{NostrRelay, RelayEvents},
//};

use bevy_async_task::AsyncTaskRunner;
use crossbeam_channel::{bounded, Receiver};

// Main function loop, runs once on startup
fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Cyberspace Navigator".into(),
                    present_mode: PresentMode::AutoVsync,
                    prevent_default_event_handling: true,
                    canvas: Some("#cyberspace-canvas".into()),
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
        .insert_resource(UniqueNotes::default()) // Set of unique notes we've received
        .insert_resource(UniqueAvatars::default()) // Set of unique avatars we've receive
        // Events work as a way to pass data between systems
        .add_event::<NostrBevyEvents>()
        .add_event::<NoteBevyEvents>()
        .add_event::<ClickedEntity>()
        // Systems are functions that run every frame
        .add_systems(Startup, (setup, draw_ui, cyberspace_websocket))
        .add_systems(
            Update,
            (
                camera_movement_system,
                cyberspace_middleware,
                rendering_entity_system,
                make_notes_children_of_avatar.run_if(on_event::<NostrBevyEvents>()),
                handle_clicks_on_entities.run_if(on_event::<ClickedEntity>()),
                handle_clicks_on_reactions.run_if(on_event::<ClickedEntity>()),
                rotate_cyberspace_entities,
            ),
        )
        .run();
}

// We start the camera at a fixed position
// far away to get a good view of the scene
fn setup(
    mut commands: Commands,
) {
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                order: 1,
                ..Default::default()
            },
            transform: Transform::from_xyz(-2000.0, 2000.0, 2000.0)
                .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            ..default()
        },
        CameraState::default(),
    ));

    // 2D Camera - For UI, with a higher order to be rendered on top
    commands.spawn(Camera2dBundle {
        camera: Camera {
            order: 2, // Rendered after the 3D camera
            ..default()
        },
        camera_2d: Camera2d {
            // don't clear on the second camera because the first camera already cleared the window
            clear_color: ClearColorConfig::None,
            ..default()
        },
        ..default()
    });
}

#[derive(Resource)]
struct DisplayText(String);

#[derive(Component, Copy, Clone)]
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
    mut camera_state: Query<(&mut CameraState, &mut Transform)>,
) {
    let sensitivity = 0.42;
    let (mut camera_state, mut camera_transform) = camera_state.single_mut();

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
struct NostrBevyEvents(String, usize);

#[derive(Event, Resource)]
struct NoteBevyEvents(CyberspaceNote);

// We use these receivers to hold the data that we receive from the websocket
#[derive(Resource, Deref)]
struct NostrReceiver(Receiver<(String, usize)>);

#[derive(Resource, Deref)]
struct NoteReceiver(Receiver<CyberspaceNote>);

const RELAY_LIST: [&str; 9] = [
    "wss://relay.arrakis.lat",
    "wss://nostr.wine",
    "wss://njump.me",
    "wss://nostr-pub.wellorder.net",
    "wss://purplerelay.com",
    "wss://relay.n057r.club",
    "wss://nostr.maximacitadel.org",
    "wss://relay.geekiam.services",
    "wss://relay.hodl.ar",
];

fn cyberspace_websocket(mut commands: Commands, mut task_executor: AsyncTaskRunner<()>) {
    let (relay_tx, relay_rx) = bounded::<(String, usize)>(100);
    let (note_tx, note_rx) = bounded::<CyberspaceNote>(500);

    task_executor.start(async move {
        let nostr_client = Client::default();
        for relay in RELAY_LIST {
            if let Err(e) = nostr_client.add_relay(relay).await {
                error!("Error adding relay: {}", e);
            }
        }

        nostr_client.connect().await;

        let kinds_iter = vec![Kind::TextNote, Kind::Reaction].into_iter();
        let filter = Filter::new().kinds(kinds_iter).limit(1000);
        nostr_client.subscribe(vec![filter]).await;

        while let Ok(msg) = nostr_client.notifications().recv().await {
            match msg {
                RelayPoolNotification::Event {
                    relay_url: _,
                    event,
                } => {
                    let cyber_note = CyberspaceNote::new(event);
                    let _ = note_tx.send(cyber_note).unwrap();
                }
                RelayPoolNotification::Message {
                    relay_url: _,
                    message,
                } => match message {
                    nostr_sdk::RelayMessage::EndOfStoredEvents(_) => {
                        info!("End of stored events");
                        let _ = relay_tx.send((format!("{:?}", message), 1)).unwrap();
                    }
                    _ => {}
                },
                RelayPoolNotification::RelayStatus {
                    relay_url: _,
                    status,
                } => {
                    info!("RelayStatus: {:?}", status);
                }
                RelayPoolNotification::Stop => {
                    info!("Stopping relay pool");
                }
                RelayPoolNotification::Shutdown => {
                    info!("Shutting down relay pool");
                }
            }
        }
    });

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

    nostr_receiver.try_iter().for_each(|(response, add)| {
        if add != 0 {
            info!("{}", response);
        }
        nostr_writer.send(NostrBevyEvents(response, add));
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

    for cyber_note in note_events.read() {
        if !avatar_set.0.contains(&cyber_note.0.pubkey) {
            avatar_set.0.insert(cyber_note.0.pubkey.clone());
            commands.spawn(AvatarBundle::new(
                &cyber_note.0,
                &mut materials,
                &mut meshes,
            ));
        }

        match cyber_note.0.kind {
            1 => {
                commands.spawn((
                    NoteBundle::new(&cyber_note.0, &mut materials, &mut meshes),
                    Kind1Note(cyber_note.0.clone()),
                    EventId(cyber_note.0.id.clone()),
                ));
            }
            7 => {
                commands.spawn((
                    NoteBundle::new(&cyber_note.0, &mut materials, &mut meshes),
                    Kind7Note(cyber_note.0.clone()),
                    TaggedEvent(cyber_note.0.tagged.clone()),
                ));
            }
            _ => {}
        }
    }
}

#[derive(Component)]
struct EventId(String);

#[derive(Component)]
struct TaggedEvent(Option<String>);

#[derive(Component)]
struct Processed;

fn make_notes_children_of_avatar(
    mut commands: Commands,
    note_query: Query<(Entity, &NotePubkey, &Transform), Without<Processed>>,
    avatar_query: Query<(Entity, &AvatarPubkey, &Transform)>,
) {
    for (avatar_entity, avatar_pubkey, avatar_transform) in avatar_query.iter() {
        for (note_entity, _note, note_transform) in note_query
            .iter()
            .filter(|(_, note, _)| note.0 == avatar_pubkey.0)
        {
            // Calculate the relative position of the note to the avatar
            let relative_position = note_transform.translation - avatar_transform.translation;
            // Set the note's new transform to this relative position
            let relative_transform = Transform::from_translation(relative_position);

            // Attach the note as a child of the avatar
            commands.entity(avatar_entity).add_child(note_entity);

            // Apply the relative transform
            commands
                .entity(note_entity)
                .insert(relative_transform)
                .insert(Processed);
        }
    }
}

#[derive(Component)]
struct CyberspaceMarker {
    pub is_active: bool,
}

impl CyberspaceMarker {
    pub fn switch(&mut self) {
        self.is_active = !self.is_active;
    }
}
impl Default for CyberspaceMarker {
    fn default() -> Self {
        CyberspaceMarker { is_active: false }
    }
}

#[derive(Component)]
struct AvatarPubkey(String);

#[derive(Component)]
struct NotePubkey(String);

use serde::{Deserialize, Serialize};
#[derive(Component, Serialize, Deserialize, Debug)]
struct AvatarMetadata {
    name: String,
    about: String,
    picture: String,
}

impl Default for AvatarMetadata {
    fn default() -> Self {
        AvatarMetadata {
            name: "".to_string(),
            about: "".to_string(),
            picture: "".to_string(),
        }
    }
}

#[derive(Component)]
struct CyberTextDetails;

#[derive(Component)]
struct CyberTextHeader;

#[derive(Component)]
struct LeftHelperText;

#[derive(Component)]
struct RightHelperText;

const DARK_PURPLE: Color = Color::rgba(0.09, 0.09, 0.15, 0.42);
const DARK_BLUE: Color = Color::rgba(0.13, 0.12, 0.25, 0.21);
const LIGHT_CORAL: Color = Color::rgba(0.95, 0.60, 0.47, 0.21);
const TERRACOTA: Color = Color::rgba(0.75, 0.38, 0.29, 0.21);
const RASPBERRY: Color = Color::rgba(0.65, 0.16, 0.34, 0.21);

fn draw_ui(mut commands: Commands, asset_server: Res<AssetServer>) {
    let cyber_heading = asset_server.load("fonts/waifus.ttf");

    let ui_top_left = commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Percent(4.2),
                left: Val::Percent(4.2),
                max_width: Val::Percent(42.0),
                max_height: Val::Percent(42.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::FlexStart,
                justify_content: JustifyContent::FlexStart,
                row_gap: Val::Px(16.8),
                column_gap: Val::Px(16.8),
                flex_wrap: FlexWrap::Wrap,
                ..default()
            },
            background_color: Color::NONE.into(),
            ..default()
        })
        .id();

    let notice_text = commands
        .spawn((
            TextBundle::from_section(
                "WELCOME TO CYBERSPACE".to_string(),
                TextStyle {
                    font_size: 42.0,
                    font: cyber_heading.clone(),
                    color: Color::WHITE,
                    ..default()
                },
            )
            .with_style(Style {
                margin: UiRect::all(Val::Px(8.4)),
                padding: UiRect::all(Val::Px(8.4)),
                align_self: AlignSelf::FlexStart,
                justify_self: JustifySelf::Start,
                ..default()
            })
            .with_background_color(DARK_PURPLE.into()),
            CyberTextHeader,
        ))
        .id();

    let main_text = commands
        .spawn((TextBundle::from_section(
            "Cyberspace is a place where you can explore the Nostr network in 3D.\n\nYou can move around using WASD and the mouse.\n\nHold right click to move the camera, and hold space to accelerate.\n\nYou can click on avatars and notes to see their content.\n\n".to_string(),
            TextStyle {
                font_size: 21.0,
                font: cyber_heading.clone(),
                color: Color::WHITE,
                ..default()
            },
        ).with_style(
            Style {
                margin: UiRect::all(Val::Px(8.4)),
                padding: UiRect::all(Val::Px(8.4)),
                flex_wrap: FlexWrap::Wrap,
                ..default()
            }
        ).with_background_color(DARK_BLUE.into()), CyberTextDetails)) // Set the alignment of the TextBundle
        .id();

    let helper_board = commands
        .spawn(NodeBundle {
            style: Style {
                display: bevy::ui::Display::Flex,
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(8.4),
                height: Val::Auto,
                ..default()
            },
            background_color: Color::NONE.into(),
            ..default()
        })
        .id();

    let empty_left_text = commands
        .spawn((
            TextBundle::from_section(
                "".to_string(),
                TextStyle {
                    font_size: 21.0,
                    font: cyber_heading.clone(),
                    color: Color::WHITE,
                    ..default()
                },
            )
            .with_style(Style {
                width: Val::Percent(65.0),
                height: Val::Auto,
                display: bevy::ui::Display::Flex,
                flex_shrink: 1.0,
                margin: UiRect::all(Val::Px(8.4)),
                padding: UiRect::all(Val::Px(8.4)),
                flex_wrap: FlexWrap::Wrap,
                ..default()
            })
            .with_background_color(Color::NONE.into()),
            LeftHelperText,
        ))
        .id();

    let empty_right_text = commands
        .spawn((
            TextBundle::from_section(
                "".to_string(),
                TextStyle {
                    font_size: 21.0,
                    font: cyber_heading.clone(),
                    color: Color::WHITE,
                    ..default()
                },
            )
            .with_style(Style {
                margin: UiRect::all(Val::Px(8.4)),
                padding: UiRect::all(Val::Px(8.4)),
                display: bevy::ui::Display::Flex,
                flex_shrink: 1.0,
                width: Val::Percent(25.0),
                height: Val::Auto,
                flex_wrap: FlexWrap::Wrap,
                ..default()
            })
            .with_background_color(Color::NONE.into()),
            RightHelperText,
        ))
        .id();

    commands
        .entity(ui_top_left)
        .push_children(&[notice_text, main_text, helper_board]);

    commands
        .entity(helper_board)
        .push_children(&[empty_left_text, empty_right_text]);
}

#[derive(Bundle)]
struct AvatarBundle {
    pub pbr: PbrBundle,
    pub pubkey: AvatarPubkey,
    pub marker: CyberspaceMarker,
    pub pickable: (On<Pointer<Over>>, On<Pointer<Out>>, On<Pointer<Click>>),
}

impl AvatarBundle {
    pub fn new(
        cyber_note: &CyberspaceNote,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) -> AvatarBundle {
        let simhash = simhash(&cyber_note.pubkey);
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
                    base_color: TERRACOTA.into(),
                    unlit: true,
                    ..default()
                }),
                transform: Transform::from_xyz(x_f32, y_f32, z_f32),
                ..Default::default()
            },
            pubkey: AvatarPubkey(cyber_note.pubkey.clone()),
            ..Default::default()
        }
    }
}

#[derive(Component)]
struct CyberspaceUI;

impl Default for AvatarBundle {
    fn default() -> Self {
        AvatarBundle {
            pbr: PbrBundle {
                ..Default::default()
            },
            pubkey: AvatarPubkey("".to_string()),
            marker: CyberspaceMarker::default(),
            pickable: (
                On::<Pointer<Over>>::target_component_mut::<Transform>(|_over, transform| {}),
                On::<Pointer<Out>>::target_component_mut::<Transform>(|_out, transform| {}),
                On::<Pointer<Click>>::send_event::<ClickedEntity>(),
            ),
        }
    }
}

#[derive(Event)]
struct ClickedEntity(Entity, Click);

impl From<ListenerInput<Pointer<Click>>> for ClickedEntity {
    fn from(event: ListenerInput<Pointer<Click>>) -> Self {
        ClickedEntity(event.target, event.event.clone())
    }
}

fn handle_clicks_on_entities(
    mut events: EventReader<ClickedEntity>,
    mut query: Query<(
        Entity,
        &mut CyberspaceMarker,
        &mut Transform,
        Option<&AvatarPubkey>,
        Option<&Kind1Note>,
    )>,
    mut ui_header: Query<
        &mut Text,
        (
            With<CyberTextHeader>,
            Without<CyberTextDetails>,
            Without<LeftHelperText>,
            Without<RightHelperText>,
        ),
    >,
    mut ui_text: Query<
        &mut Text,
        (
            With<CyberTextDetails>,
            Without<CyberTextHeader>,
            Without<LeftHelperText>,
            Without<RightHelperText>,
        ),
    >,
    mut ui_left: Query<
        (&mut Text, &mut BackgroundColor),
        (
            With<LeftHelperText>,
            Without<CyberTextHeader>,
            Without<CyberTextDetails>,
            Without<RightHelperText>,
        ),
    >,
    mut ui_right: Query<
        (&mut Text, &mut BackgroundColor),
        (
            With<RightHelperText>,
            Without<CyberTextHeader>,
            Without<CyberTextDetails>,
            Without<LeftHelperText>,
        ),
    >,
) {
    let mut ui_text = ui_text.single_mut();
    let mut ui_header = ui_header.single_mut();
    let (mut ui_left, mut ui_left_bg) = ui_left.single_mut();
    let (mut ui_right, mut ui_right_bg) = ui_right.single_mut();

    for event in events.read() {
        if let Ok((entity, mut marker, mut transform, avatar_pubkey, note)) = query.get_mut(event.0)
        {
            match event.1.button {
                PointerButton::Secondary => {
                    if note.is_some() {
                        transform.scale = if marker.is_active {
                            Vec3::splat(1.0)
                        } else {
                            Vec3::splat(2.1)
                        };
                        marker.switch();
                    }
                    if avatar_pubkey.is_some() {
                        transform.scale = if marker.is_active {
                            Vec3::splat(1.0)
                        } else {
                            Vec3::splat(4.2)
                        };
                        marker.switch();
                    }
                }
                PointerButton::Primary => {
                    if let Some(avatar) = avatar_pubkey {
                        ui_header.sections[0].value = format!("AVATAR");
                        ui_text.linebreak_behavior = BreakLineOn::AnyCharacter;
                        ui_text.sections[0].value = format!("{}", avatar.0);
                        ui_left.sections[0].value = format!("");
                        ui_left_bg.0 = Color::NONE.into();
                        ui_right.sections[0].value = format!("");
                        ui_right_bg.0 = Color::NONE.into();
                    } else if let Some(note) = note {
                        let options = Options::new(42)
                            .word_separator(WordSeparator::UnicodeBreakProperties)
                            .break_words(true);

                        let wrapped_content = wrap(&note.0.content, options);

                        ui_header.sections[0].value = format!("NOTE");
                        ui_text.linebreak_behavior = BreakLineOn::AnyCharacter;
                        ui_text.sections[0].value = format!("{}", wrapped_content.join("\n"));
                        ui_left.linebreak_behavior = BreakLineOn::AnyCharacter;
                        ui_left.sections[0].value = format!("{}", note.0.created_at);
                        ui_left_bg.0 = RASPBERRY.into();
                        ui_right.linebreak_behavior = BreakLineOn::AnyCharacter;
                        ui_right.sections[0].value = format!("{}", note.0.id[..12].to_string());
                        ui_right_bg.0 = DARK_BLUE.into();
                    }
                }
                _ => {}
            }
        }
    }
}

fn handle_clicks_on_reactions(
    mut events: EventReader<ClickedEntity>,
    mut query: Query<(&TaggedEvent, &Kind7Note)>,
    mut ids_query: Query<(&EventId, &Kind1Note)>,
    mut ui_header: Query<
        &mut Text,
        (
            With<CyberTextHeader>,
            Without<CyberTextDetails>,
            Without<LeftHelperText>,
            Without<RightHelperText>,
        ),
    >,
    mut ui_text: Query<
        &mut Text,
        (
            With<CyberTextDetails>,
            Without<CyberTextHeader>,
            Without<LeftHelperText>,
            Without<RightHelperText>,
        ),
    >,
    mut ui_left: Query<
        (&mut Text, &mut BackgroundColor),
        (
            With<LeftHelperText>,
            Without<CyberTextHeader>,
            Without<CyberTextDetails>,
            Without<RightHelperText>,
        ),
    >,
    mut ui_right: Query<
        (&mut Text, &mut BackgroundColor),
        (
            With<RightHelperText>,
            Without<CyberTextHeader>,
            Without<CyberTextDetails>,
            Without<LeftHelperText>,
        ),
    >,
) {
    let mut ui_text = ui_text.single_mut();
    let mut ui_header = ui_header.single_mut();
    let (mut ui_left, mut ui_left_bg) = ui_left.single_mut();
    let (mut ui_right, mut ui_right_bg) = ui_right.single_mut();

    for event in events.read() {
        if let Ok((tagged_event, _)) = query.get_mut(event.0) {
            match event.1.button {
                PointerButton::Primary => {
                    if let Some(tag) = &tagged_event.0 {
                        for (id, note) in ids_query.iter_mut() {
                            if id.0 == tag.clone() {
                                info!("Found the note: {}", note.0);
                                ui_header.sections[0].value = format!("REACTION");
                                ui_text.linebreak_behavior = BreakLineOn::AnyCharacter;
                                let options = Options::new(42)
                                    .word_separator(WordSeparator::UnicodeBreakProperties)
                                    .break_words(true);

                                let wrapped_content = wrap(&note.0.content, options);

                                ui_text.sections[0].value =
                                    format!("{}", wrapped_content.join("\n"));
                                ui_left.linebreak_behavior = BreakLineOn::AnyCharacter;
                                ui_left.sections[0].value = format!("{}", note.0.created_at,);
                                ui_left_bg.0 = RASPBERRY.into();
                                ui_right.linebreak_behavior = BreakLineOn::AnyCharacter;
                                ui_right.sections[0].value =
                                    format!("{}", note.0.id[..12].to_string());
                                ui_right_bg.0 = DARK_BLUE.into();
                            }
                        }
                    } else {
                        ui_header.sections[0].value = format!("Reacted To:");
                        ui_text.sections[0].value = format!("Not found in this relay galaxy");
                    }
                }
                _ => {}
            }
        }
    }
}

#[derive(Component, Debug)]
struct Kind1Note(CyberspaceNote);

#[derive(Component, Debug)]
struct Kind7Note(CyberspaceNote);

#[derive(Bundle)]
struct NoteBundle {
    pub pbr: PbrBundle,
    pub note_pubkey: NotePubkey,
    pub marker: CyberspaceMarker,
}

impl Default for NoteBundle {
    fn default() -> Self {
        NoteBundle {
            pbr: PbrBundle {
                ..Default::default()
            },
            note_pubkey: NotePubkey("".to_string()),
            marker: CyberspaceMarker::default(),
        }
    }
}

impl NoteBundle {
    pub fn new(
        note: &CyberspaceNote,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        meshes: &mut ResMut<Assets<Mesh>>,
    ) -> NoteBundle {
        let hash: U256;
        let origin_hash: U256;
        let (x, y, z);
        let color;
        let radius;

        match note.kind {
            1 => {
                hash = simhash(&note.content);
                origin_hash = simhash(&note.pubkey);
                (x, y, z) = map_hash_to_coordinates_with_offset(hash, origin_hash);
                color = RASPBERRY.into();
                radius = 2.1;
            }
            7 => {
                let concat_str = format!("{} {} {}", note.content, note.created_at, note.id);
                hash = simhash(&concat_str);
                origin_hash = simhash(&note.pubkey);
                (x, y, z) = map_hash_to_coordinates_with_offset(hash, origin_hash);
                color = DARK_PURPLE.into();
                radius = 1.05;
            }
            _ => {
                (x, y, z) = (0.0, 0.0, 0.0);
                color = Color::rgba(1.0, 1.0, 1.0, 1.0);
                radius = 0.0;
            }
        }
        NoteBundle {
            pbr: PbrBundle {
                mesh: meshes.add(
                    Mesh::try_from(shape::Icosphere {
                        radius,
                        subdivisions: 4,
                        ..Default::default()
                    })
                    .unwrap(),
                ),
                material: materials.add(StandardMaterial {
                    base_color: color,
                    unlit: true,
                    ..default()
                }),
                transform: Transform::from_xyz(x, y, z),
                ..Default::default()
            },
            note_pubkey: NotePubkey(note.pubkey.clone()),
            ..Default::default()
        }
    }
}

#[derive(Event, Debug, Clone)]
struct CyberspaceNote {
    content: String,
    kind: u32,
    id: String,
    created_at: String,
    pubkey: String,
    tagged: Option<String>,
}

impl CyberspaceNote {
    pub fn new(signed_note: nostr_sdk::Event) -> CyberspaceNote {
        let content = signed_note.content.clone();
        let kind = signed_note.kind.as_u32();
        let id = signed_note.id.to_hex();
        let created_at = signed_note.created_at.to_human_datetime();
        let pubkey = signed_note.pubkey.to_string();
        let tagged = signed_note.event_ids().map(|id| id.to_hex()).next();

        CyberspaceNote {
            content,
            kind,
            id,
            created_at,
            pubkey,
            tagged,
        }
    }
}

use textwrap::{wrap, Options, WordSeparator};

impl Display for CyberspaceNote {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let options = Options::new(420)
            .word_separator(WordSeparator::AsciiSpace)
            .break_words(true);

        let wrapped_content = wrap(&self.content, options);

        write!(
            f,
            "{},\n\nTimestamp: {},\n\nID {}..,\n",
            wrapped_content.join("\n"),
            self.created_at,
            self.id[..8].to_string(),
        )
    }
}

// CYBERSPACE METHODS
// These methods are used to generate the cyberspace coordinates for the notes and avatars
// based on their content and public key respectively

// Simhash is a hashing algorithm that is used to generate a 256 bit hash from a string
fn simhash(input: &str) -> U256 {
    // Initialize a vector of 256 zeros
    let mut vectors: Vec<i32> = vec![0; 256];

    // Initialize a hashmap to count the occurrences of each shingle
    let mut shingle_count = HashMap::new();

    // Convert input to a vector of characters
    let chars: Vec<char> = input.chars().collect();

    if chars.len() > 1 {
        for i in 0..chars.len() - 1 {
            let shingle = chars[i].to_string() + &chars[i + 1].to_string();
            *shingle_count.entry(shingle).or_insert(0) += 1;
        }
    }
    // Hash each shingle and add/subtract from the vector
    for (shingle, count) in shingle_count {
        // Hash the shingle
        let hash = hash_word(&shingle); // Assuming hash_word can hash a shingle
        for i in 0..256 {
            // Add or subtract from the vector based on the bit at index i
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

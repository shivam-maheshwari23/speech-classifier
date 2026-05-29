COMMANDS: list[dict] = [
    {
        "label": "activate_do_not_disturb",
        "phrases": [
            "activate do not disturb",
            "turn on do not disturb",
            "enable do not disturb",
            "switch on do not disturb",
            "put on do not disturb",
            "start do not disturb mode",
            "do not disturb on",
            "silence notifications",
        ]
    },
    {
        "label": "deactivate_do_not_disturb",
        "phrases": [
            "deactivate do not disturb",
            "turn off do not disturb",
            "disable do not disturb",
            "switch off do not disturb",
            "stop do not disturb mode",
            "do not disturb off",
            "allow notifications",
        ]
    },
    {
        "label": "decline_call",
        "phrases": [
            "decline the call",
            "reject the call",
            "ignore the call",
            "don't answer",
            "reject call",
            "hang up",
            "dismiss the call",
        ]
    },
    {
        "label": "pick_up_call",
        "phrases": [
            "pick up the call",
            "answer the call",
            "accept the call",
            "take the call",
            "receive the call",
            "answer call",
            "pick up",
        ]
    },
    {
        "label": "play_music",
        "phrases": [
            "play the music",
            "start the music",
            "play music",
            "resume music",
            "start music",
            "play some music",
            "turn on music",
        ]
    },
    {
        "label": "pause_music",
        "phrases": [
            "pause the music",
            "stop the music",
            "pause music",
            "hold the music",
            "mute music",
            "stop music",
            "pause the song",
        ]
    },
    {
        "label": "play_next_song",
        "phrases": [
            "play the next song",
            "next song",
            "skip this song",
            "next track",
            "skip track",
            "forward song",
            "play next",
            "next track please",
        ]
    },
    {
        "label": "play_previous_song",
        "phrases": [
            "play the previous song",
            "previous song",
            "go back a song",
            "previous track",
            "last song",
            "play previous",
            "back track",
            "go back a track",
            "go back",
        ]
    },
    {
        "label": "increase_volume",
        "phrases": [
            "increase the volume",
            "turn up the volume",
            "volume up",
            "louder",
            "raise the volume",
            "make it louder",
            "turn it up",
            "increase volume",
        ]
    },
    {
        "label": "decrease_volume",
        "phrases": [
            "decrease the volume",
            "turn down the volume",
            "volume down",
            "quieter",
            "lower the volume",
            "make it quieter",
            "turn it down",
            "decrease volume",
        ]
    },
    {
        "label": "increase_brightness",
        "phrases": [
            "increase the brightness",
            "turn up the brightness",
            "brightness up",
            "make it brighter",
            "raise the brightness",
            "higher brightness",
            "increase brightness",
        ]
    },
    {
        "label": "decrease_brightness",
        "phrases": [
            "decrease the brightness",
            "turn down the brightness",
            "brightness down",
            "make it dimmer",
            "lower the brightness",
            "lower brightness",
            "decrease brightness",
        ]
    },
    {
        "label": "start_vehicle",
        "phrases": [
            "start the vehicle",
            "start the car",
            "turn on the car",
            "engine on",
            "start engine",
            "ignition on",
            "turn the car on",
        ]
    },
    {
        "label": "stop_vehicle",
        "phrases": [
            "stop the vehicle",
            "stop the car",
            "turn off the car",
            "engine off",
            "stop engine",
            "ignition off",
            "turn the car off",
        ]
    },
]

OUT_OF_SCOPE: list[str] = [
    "what is the weather today",
    "navigate to the nearest petrol station",
    "call John",
    "hello",
    "turn off the air conditioning",
    "what time is it",
    "open google maps",
    "set an alarm",
    "how far is the next exit",
    "send a message to mom",
]


def get_training_data() -> tuple[list[str], list[str]]:
    """Returns all phrases and their labels for training."""
    texts: list[str] = []
    labels: list[str] = []
    for command in COMMANDS:
        for phrase in command["phrases"]:
            texts.append(phrase)
            labels.append(command["label"])
    return texts, labels


def get_all_labels() -> list[str]:
    """Returns list of all command label names."""
    return [command["label"] for command in COMMANDS]
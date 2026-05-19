"""
narrative_dictionary.py — The voice of Chin Music.

A library of broadcast-style commentary strings for the game engine.
Keys map to PA outcome types, pitching events, and situational calls.
Each key holds at least 5 variants so the output never sounds robotic.

Usage:
    from narrative_dictionary import narrate

    narrate("HOME_RUN", "Babe Ruth")
    # → "Babe Ruth crushes it. Home run."

    narrate("RUNS_SCORE", count=3)
    # → "3 cross the plate."

    narrate("PITCHER_TIRED", "Pedro Martinez")
    # → "The velocity is dipping for Pedro Martinez."
"""

import random
from typing import Final


NARRATIVE_TEMPLATES: Final[dict[str, list[str]]] = {

    # ── Batter outcomes ──────────────────────────────────────────────────────

    "STRIKEOUT": [
        "{name} goes down swinging.",
        "{name} can't hold off — strikeout.",
        "{name} freezes on a called strike three.",
        "Swing and a miss — {name} fans.",
        "{name} chases it out of the zone. Punch out.",
        "Three up, three strikes. {name} walks back to the dugout.",
        "{name} is fooled completely. Strikeout.",
        "Punchout. {name} couldn't make contact.",
        "{name} takes a big cut and misses. He's done.",
        "{name} rings up. End of at-bat.",
    ],

    "WALK": [
        "{name} works a walk.",
        "Ball four. {name} takes first.",
        "{name} earns the free pass.",
        "Four wide ones — {name} draws the walk.",
        "{name} battles and gets a walk.",
        "The pitcher can't find the zone. {name} walks.",
        "{name} lays off the breaking ball. Ball four.",
        "{name} has a good eye tonight — walk.",
    ],

    "HIT_BY_PITCH": [
        "{name} is plunked. Takes first.",
        "Inside. That catches {name} — HBP.",
        "{name} absorbs it and heads to first.",
        "That clipped {name}. He'll take his base.",
        "Hit by pitch. {name} trots to first.",
    ],

    "SINGLE": [
        "{name} singles through the left side.",
        "{name} punches one into right field.",
        "Base hit — {name} finds a gap.",
        "{name} lines it into shallow center. Single.",
        "A clean single for {name}.",
        "{name} places one just out of the second baseman's reach.",
        "{name} drops one into right-center. He's on.",
        "A well-placed hit by {name}.",
        "{name} gets the barrel on it — base hit.",
        "{name} slaps one through the hole. Single.",
    ],

    "DOUBLE": [
        "{name} cracks one into the gap — double.",
        "{name} doubles down the line.",
        "Off the wall! {name} pulls into second.",
        "{name} drives it deep into the corner.",
        "Extra-base knock for {name} — two-bagger.",
        "{name} lashes it to left-center. He'll take two.",
        "The outfielder gives chase but can't get there. {name} has a double.",
        "{name} scalds one down the right-field line. Double.",
    ],

    "TRIPLE": [
        "{name} legs it out all the way to third!",
        "{name} rips one into the deep corner — triple!",
        "The outfield can't track it down. {name} triples.",
        "{name} turns on the jets — he's standing at third.",
        "A long drive off the bat of {name}. He takes three!",
        "They're not even throwing home — it's a triple for {name}.",
    ],

    "HOME_RUN": [
        "{name} puts a charge into one — that ball is gone.",
        "{name} crushes it. Home run.",
        "Back, back, back — {name} hits it out.",
        "{name} launches one well into the seats.",
        "That ball is not coming back. Home run, {name}.",
        "{name} connects — and that's a homer.",
        "Long fly ball … gone. {name} takes a leisurely trip around the bases.",
        "{name} did not miss that one. Way out.",
        "Gone! {name} just hit one into the upper deck.",
        "Deep fly ball … it's outta here! {name} goes yard.",
    ],

    "OUT": [
        "{name} grounds out.",
        "{name} flies out to the warning track.",
        "A lazy fly ball off the bat of {name} — caught.",
        "{name} ropes it right at the shortstop. Out.",
        "Routine groundout. {name} is thrown out at first.",
        "{name} pops it up — easy out.",
        "{name} lines it to the infield. Caught for the out.",
        "Nice play by the third baseman on {name}.",
        "{name} hits it on the screws — right at the center fielder.",
        "{name} bounces one to second. One away.",
    ],

    "REACHED_ON_ERROR": [
        "Misplay in the field. {name} reaches on an error.",
        "The fielder boots it. {name} is safe.",
        "A bobble in the infield keeps {name} alive.",
        "The throw pulls him off the bag — error. {name} is safe.",
        "He's safe! The defense gives {name} a gift.",
        "Should have been an out. Error on the play. {name} is aboard.",
    ],

    # ── Pitching events ──────────────────────────────────────────────────────

    "PITCHER_STARTS": [
        "{name} takes the mound.",
        "{name} gets the ball today.",
        "{name} is on the hill to start.",
        "First pitch goes to {name}.",
        "{name} toes the rubber. Let's go.",
    ],

    "PITCHER_CHANGE": [
        "{name} trots in from the bullpen.",
        "{name} gets the ball. New pitcher.",
        "The manager goes to the mound — {name} is in.",
        "{name} coming on in relief.",
        "The bullpen door opens. {name} is the new pitcher.",
        "Here comes {name} from the pen.",
        "{name} takes over on the hill.",
    ],

    "PITCHER_TIRED": [
        "{name} is losing his rhythm.",
        "{name} looks gassed out there.",
        "The velocity is dipping for {name}.",
        "The manager is getting antsy — {name} is laboring.",
        "{name} has been out there a long time.",
        "The effort is showing. {name} is running out of gas.",
        "{name} is grinding through it. The bullpen is warming.",
    ],

    # ── Run scoring ──────────────────────────────────────────────────────────

    "RUN_SCORES": [
        "A run comes home.",
        "That scores one.",
        "The runner crosses the plate.",
        "One comes in.",
        "A run scores.",
        "He scores easily.",
    ],

    "RUNS_SCORE": [
        "{count} come around to score.",
        "{count} runs score on the play.",
        "{count} cross the plate.",
        "A big play — {count} runs in.",
        "They're pouring it on: {count} runs score.",
    ],

    # ── Baserunning results ──────────────────────────────────────────────────

    "ADVANCE_EXTRA": [
        "{name} never stops — he's in at third!",
        "{name} reads it perfectly and takes the extra base.",
        "{name} rounds the bag and doesn't look back.",
        "The outfielder bobbles it and {name} takes third.",
        "{name} turns on the jets — extra base!",
        "{name} sends himself. He's in there!",
    ],

    "THROWN_OUT_BASES": [
        "{name} tries for the extra base — gunned down!",
        "Perfect throw — {name} is out trying to advance.",
        "{name} gets thrown out. He read that wrong.",
        "The relay is on the money. {name} is out.",
        "{name} never had a chance. Out.",
        "They get him. {name} is erased on the bases.",
    ],

    "THROWN_OUT_AT_HOME": [
        "Out at home! {name} is cut down at the plate.",
        "Perfect throw — {name} is nailed at home.",
        "The relay beats him. {name} is out at home.",
        "{name} was never going to make it. Out at the plate.",
        "The catcher blocks the plate — {name} is out!",
        "{name} goes for it, but the throw has him. Out at home.",
    ],

    "THROWN_OUT_AT_THIRD": [
        "{name} is cut down at third!",
        "They throw him out at third. {name} had no chance.",
        "Perfect relay to the bag — {name} is out at third.",
        "{name} tries for third and they nail him.",
        "The third baseman has it — out! {name} is done.",
        "The arm from the outfield catches {name} at third.",
    ],

    "DOUBLE_PLAY": [
        "{name} hits into a double play.",
        "6-4-3 double play. Two down.",
        "{name} grounds into the twin killing.",
        "They turn two on {name}. Inning-killer.",
        "Double play. The defense bails out the pitcher.",
        "{name} bounces into a double play. Quick two outs.",
        "Hard contact right to the second baseman — 4-6-3 double play.",
        "{name} scorched it, but right at the shortstop. Double play.",
    ],

    "INFIELD_HIT": [
        "Slow roller — no play at first. {name} beats it out.",
        "{name} legs it out on a slow chopper. Infield single.",
        "The third baseman had to charge — not in time. {name} is safe.",
        "Nobody had a play. {name} reaches on an infield hit.",
        "{name} muscles it through the right side — infield single.",
        "{name} beats it out with pure speed.",
        "Sneaky infield hit for {name}.",
        "The ball dies on the grass. {name} is safe at first.",
        "{name} hustles down the line. Safe!",
        "No chance for the throw — {name} reaches on an infield hit.",
    ],

    "WEAK_ROLLER_NO_DP": [
        "It was a slow roller — they can only get the one at first.",
        "The weak contact saved them from the double play.",
        "Soft off the bat of {name} — not enough to turn two.",
        "A dying quail grounder. {name} is out, but the runner is safe.",
        "Too slow off the bat. The second baseman can only take the sure out.",
        "{name} barely hit it. No chance for two.",
        "The infield had to charge — no time to throw to second first.",
    ],

    "SMART_HOLD": [
        "Smart read — {name} holds at second.",
        "{name} gets the stop sign and takes it. Good decision.",
        "The coach holds {name} up. Right call.",
        "{name} doesn't challenge that arm. He holds.",
        "Heads-up baserunning from {name} — stays put.",
    ],

    # ── Situational / atmosphere ─────────────────────────────────────────────

    "RECKLESS_ADVANCE": [
        "{name} ignored the stop sign!",
        "A mental error by {name} on the bases.",
        "{name} tried to force it and paid the price.",
        "{name} never should have tried for the extra base.",
        "The third base coach waved him — and {name} was dead to rights.",
        "You don't run on that arm. {name} did. Big mistake.",
    ],

    "WALK_OFF": [
        "Ball game. Walk-off.",
        "{name} wins it right here.",
        "The crowd erupts — it's over.",
        "Walk-off. Ball game.",
        "They don't need the last out. {name} ends it.",
        "And that's the ballgame! Walk-off win.",
    ],
}


def narrate(key: str, name: str = "", **kwargs) -> str:
    """
    Pick a random variant for *key* and fill in {name} and any **kwargs.

    Args:
        key:   One of the keys in NARRATIVE_TEMPLATES.
        name:  The primary player name to inject (maps to {name} placeholder).
        **kwargs: Any additional template variables (e.g. count=3 for RUNS_SCORE).

    Returns:
        A formatted narrative string.
        Falls back to "[KEY]" if the key is not found in the dictionary.
    """
    templates = NARRATIVE_TEMPLATES.get(key)
    if not templates:
        return f"[{key}]"
    return random.choice(templates).format(name=name, **kwargs)

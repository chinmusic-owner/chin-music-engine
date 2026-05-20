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
        "{name} strikes out swinging.",
        "{name} goes down on strikes.",
        "{name} strikes out.",
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
        "Chin music. {name} gets plunked — takes first.",
        "Look out! {name} wears it. He'll take his base.",
        "Watch out! That one catches {name} on the arm. HBP.",
        "He wears it. {name} grimaces and trots to first.",
        "Right in the ribs — {name} takes first.",
        "Ooh. {name} is hit. He shakes it off and heads to first.",
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

    # Situational HR variants — chosen by print_game_log based on context.
    # These are never picked by the generic narrate("HOME_RUN") path.

    "HOME_RUN_SOLO_LATE": [
        "That one had some juice. {name} goes deep.",
        "{name} hits it with something on it. Gone. Late-game solo shot.",
        "Big swing from {name} — that ball is not coming back.",
        "{name} gets all of it. Solo homer, late in the game.",
        "A towering fly ball by {name}. Gone.",
    ],

    "HOME_RUN_GOAHEAD": [
        "That's the lead. {name} puts {team} on top.",
        "{name} delivers the go-ahead home run.",
        "Go-ahead shot from {name}. {team} in front.",
        "{team} takes the lead. {name} made that look easy.",
        "{name} just flipped this game. {team} on top.",
    ],

    "HOME_RUN_TIEBREAKER": [
        "{name} just changed the game.",
        "Late-game heroics. {name} delivers the go-ahead shot.",
        "You could feel it coming. {name} puts {team} in front.",
        "That's the moment right there. {name} goes deep when it matters.",
        "{name} hits it when this game needed it most.",
    ],

    "HOME_RUN_TWO_RUN": [
        "{name} goes deep — {runner1} scores ahead of him.",
        "Two-run blast by {name}. {runner1} and {name} come home.",
        "{name} drives one out. {runner1} scores on the play.",
        "Gone. Two runs score. {runner1} leads the way.",
        "{name} doesn't miss it. {runner1} scores. Two-run homer.",
    ],

    "HOME_RUN_THREE_RUN": [
        "{name} goes yard with two on — {runner1} and {runner2} come home.",
        "Three-run blast by {name}. {runner1} and {runner2} score.",
        "{name} launches one. {runner1} and {runner2} cross the plate.",
        "{runner1}, {runner2}, and {name} all score. Three-run homer.",
        "{name} goes deep with two aboard — {runner1} and {runner2} come around.",
    ],

    "HOME_RUN_GRAND_SLAM": [
        "Grand slam! {name} clears the bases.",
        "{name} hits the grand salami. Four runs score.",
        "The bases were loaded — and {name} just unloaded. Grand slam!",
        "Four runs score. {name} with the grand slam.",
        "{name} with the bases loaded — gone! Grand slam!",
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
        "The velocity is dipping for {name}.",
        "The manager is getting antsy — {name} is laboring.",
        "{name} has been out there a long time.",
        "The effort is showing. {name} is running out of gas.",
        "{name} is grinding through it. The bullpen is warming.",
        "The zone is leaking. {name} is working on fumes.",
    ],

    "PITCHER_GASSED": [
        "{name} has nothing left. He's running on empty.",
        "The plate got small for {name}. He's done.",
        "{name} is out of gas — the bullpen needs to move.",
        "Heavy arm. {name} can't finish his pitches.",
        "{name} looks cooked. Every batter is a danger now.",
        "The manager has seen enough. {name} is gassed.",
    ],

    "PITCHER_COLLAPSE": [
        "The damage comes when a pitcher has nothing left. {name} is paying for it.",
        "{name} gave up a run he can't afford. The bullpen is up.",
        "Hard contact against a tired arm. {name} is getting punished.",
        "The zone leaked and the offense cashed it. {name} takes the hit.",
        "The arm is empty — {name} is leaving pitches right where hitters want them.",
    ],

    # Collapse narrative specific to walks — "hard contact" language is wrong
    # when the run scored via a base on balls.
    "PITCHER_COLLAPSE_WALK": [
        "{name} can't find the zone. The walks are adding up.",
        "Losing the plate. {name} is handing out free passes.",
        "{name} has no command left — ball four again.",
        "The control is gone. {name} is walking them home.",
        "Another walk. {name} is leaking badly.",
    ],

    "PITCHING_CHANGE": [
        "{name} gets the ball. The starter is done.",
        "The manager goes to the mound — {name} is in.",
        "{name} trots in from the bullpen.",
        "{name} coming on in relief. New arm, same pressure.",
        "The bullpen door opens. {name} inherits the situation.",
        "Here comes {name} from the pen. The game is in his hands.",
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

    # Primary play-by-play line for a double play — replaces the generic contact
    # template so the main narrative is DP-specific and the ↳ sub-line is suppressed.
    "DOUBLE_PLAY_PRIMARY": [
        "{name} bounces one to short — they turn two.",
        "{name} hits a sharp grounder. Around the horn — double play.",
        "6-4-3 on {name}. Two down.",
        "{name} tops one to second — twin killing.",
        "Chopper off the bat of {name}. They get two.",
        "{name} grounds it right at the shortstop. Double play.",
        "A tailor-made double play off the bat of {name}.",
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
        "Too slow off the bat. The second baseman can only take the sure out.",
        "{name} barely hit it. No chance for two.",
        "The infield had to charge — no time to throw to second first.",
        "They couldn't set up for two. {name} is out at first, runner stays.",
    ],

    "FIELDERS_CHOICE": [
        "{runner_out} retired at second — {name} safe at first. Fielder's choice.",
        "Grounder to the infield — {runner_out} thrown out at second. {name} reaches first.",
        "Force at second on {runner_out}. {name} beats it out at first.",
        "Sharp grounder — {runner_out} out at second. {name} is safe. Fielder's choice.",
        "They go to second for {runner_out}. {name} beats the throw. Fielder's choice.",
        "The fielder fires to second — {runner_out} is out. {name} safe at first.",
        "Chopper to the infield. {runner_out} forced at second. {name} reaches on the fielder's choice.",
    ],

    "GROUNDER_WEAK": [
        "{name} taps it softly — thrown out at first.",
        "A weak dribbler from {name}. Easy play. Out at first.",
        "{name} barely gets it off the bat. Out at first.",
        "Soft contact from {name} — fielder has all the time in the world. Out.",
        "{name} rolls it over to the right side. Throw beats him at first.",
        "{name} punches a weak grounder — out at first.",
    ],

    "GROUNDER_HARD": [
        "{name} smokes it right at the shortstop — throw to first. Out.",
        "Hard shot by {name} — right at the third baseman. Out.",
        "{name} scorches one right at the second baseman. Out.",
        "Rocket off the bat of {name} — snagged on a short hop. Out at first.",
        "{name} lines it hard. Gloved, thrown out at first.",
        "{name} crushes it — right at someone. Out.",
    ],

    # ── Fielder-named out templates (FIX 4) ─────────────────────────────────
    # All accept {name} (batter), {fielder} (defender), {pos} (position label).

    "OUT_FLY": [
        "{name} sends one to {pos} — {fielder} tracks it down.",
        "{name} lifts a fly to {pos}. {fielder} settles under it. Easy out.",
        "Deep drive by {name} — {fielder} runs it down at the track.",
        "High fly to {pos} — {fielder} hauls it in.",
        "{fielder} comes over for it in {pos}. Catches it. One out.",
    ],

    "OUT_GROUNDER": [
        "{name} bounces one to {fielder} at {pos} — throw to first. Out.",
        "{name} chops one to {fielder} — gets him by a step at first.",
        "{fielder} charges at {pos} and fires to first. {name} is out.",
        "Sharp chopper to {fielder} — long throw, got him.",
        "{name} rolls one to {fielder} at {pos}. Out by a step.",
    ],

    "OUT_POPUP": [
        "{name} pops it up — {fielder} calls off everyone. Routine.",
        "{name} pops it up — {fielder} squeezes it. Out.",
        "{name} lifts a high pop-up. {fielder} camps under it. Out.",
        "{name} gets under one — {fielder} settles under it. Easy out.",
        "{name} pops it up to {fielder}. Routine catch.",
    ],

    "OUT_LINER": [
        "{name} scorches one — {fielder} snags it.",
        "{name} lines it right at {fielder} in {pos}. No chance.",
        "Smoking liner from {name} — right at {fielder}. Out.",
        "{name} rips it, but {fielder} was there. Caught.",
        "Shot off the bat of {name} — {fielder} spears it in {pos}.",
    ],

    "OUT_COMEBACKER": [
        "{name} taps it back — {fielder} fields it and flips to first.",
        "Soft comebacker from {name}. {fielder} has all day. Out.",
        "{name} rolls one back to the mound — {fielder} fields and fires. Out.",
        "Easy play for {fielder} — {name} bounced it right back. Out.",
        "{name} sends it back to the mound. {fielder} fires to first. Out.",
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


# ── Grounder-out variant pool ────────────────────────────────────────────────────
# Each entry: (template_string, weight).  Higher weight = chosen more often.
# {name}=batter, {fielder}=defender, {pos}=display position label.
#
# Position groups:
#   "long"    → 3B / SS (cross-diamond throw — distance matters)
#   "flip"    → 2B      (short flip or pivot throw)
#   "self"    → 1B      (fielder takes it himself at the bag)
#   "neutral" → any     (merged into every group as low-weight fallbacks)
GROUNDER_POOL: dict[str, list[tuple[str, int]]] = {
    "long": [
        ("{name} bounces one to {fielder} at {pos} — long throw, gets him.", 3),
        ("{name} chops one to {fielder} — long throw across the diamond, out.", 2),
        ("{fielder} charges, fields it clean, fires to first. {name} is out.", 2),
        ("{name} rolls one to {fielder} — good arm gets him at first.", 2),
        ("{name} taps it toward {pos} — {fielder} has to hurry. Gets him.", 2),
        ("Slow roller to {fielder} at {pos}. He charges and fires. Out.", 1),
    ],
    "flip": [
        ("{name} bounces one to {fielder} at {pos}. Quick flip to first.", 3),
        ("{name} rolls one toward second — {fielder} flips to first. Out.", 3),
        ("Easy chance for {fielder} — {name} is out at first.", 2),
        ("{name} taps it softly to {fielder}. He flips over. Out at first.", 2),
        ("{name} chops one to {fielder} — short flip and he's got him.", 2),
    ],
    "self": [
        ("{fielder} takes the throw at first — {name} is out.", 3),
        ("{name} rolls one to the right side — {fielder} steps on the bag. Out.", 3),
        ("{name} taps it to {fielder} — he steps on first. Out.", 2),
        ("Soft grounder to {fielder} — he beats {name} to the bag. Out.", 2),
        ("{name} bounces one to {fielder} at first. {fielder} handles it himself. Out.", 2),
    ],
    "neutral": [
        ("{name} hits a slow roller to {fielder}. Plenty of time. Out at first.", 3),
        ("{name} bounces one to {fielder} — routine play. Out at first.", 3),
        ("Routine grounder to {fielder} — throw to first. Out.", 3),
        ("{name} taps it softly to {fielder} — throw to first. In time.", 2),
        ("{name} rolls one to {fielder} — fields it and fires. Out.", 2),
        ("{name} sends a chopper to {fielder} — gets him at first.", 2),
        ("{name} hits it on the ground to {fielder}. Throw to first. Out.", 2),
        ("{name} chops one to {fielder} — throws him out at first.", 2),
    ],
}

# Maps infield position code → GROUNDER_POOL group
_GROUNDER_POS_GROUP: dict[str, str] = {
    "3B": "long",
    "SS": "long",
    "2B": "flip",
    "1B": "self",
}


def narrate_grounder(
    name: str,
    fielder: str,
    pos_code: str,
    pos_label: str,
    history: list,
) -> str:
    """
    Pick a position-aware grounder narrative, avoiding the last 2 used templates.

    name      : batter name
    fielder   : defender's name
    pos_code  : defensive position code — "3B", "SS", "2B", "1B"
    pos_label : display label passed to {pos} slot ("third base", "short", …)
    history   : mutable list tracking the last 2 used template strings (caller-owned)
    """
    group = _GROUNDER_POS_GROUP.get(pos_code, "neutral")

    # Build candidate pool: position-specific entries at 2× weight, then neutral at 1×.
    seen_tmpl: set[str] = set()
    candidates: list[tuple[str, int]] = []

    for tmpl, weight in GROUNDER_POOL.get(group, []):
        if tmpl not in seen_tmpl:
            seen_tmpl.add(tmpl)
            candidates.append((tmpl, weight * 2))   # position-specific: 2× boost

    for tmpl, weight in GROUNDER_POOL["neutral"]:
        if tmpl not in seen_tmpl:
            seen_tmpl.add(tmpl)
            candidates.append((tmpl, weight))        # neutral: base weight

    # Filter out templates used in the last 2 at-bats
    available = [(t, w) for t, w in candidates if t not in history]
    if not available:
        available = candidates                        # all filtered? relax history

    # Weighted random selection
    total = sum(w for _, w in available)
    roll  = random.random() * total
    cumulative = 0.0
    chosen = available[-1][0]
    for tmpl, weight in available:
        cumulative += weight
        if roll <= cumulative:
            chosen = tmpl
            break

    # Update caller's history (keep last 2)
    history.append(chosen)
    if len(history) > 2:
        history.pop(0)

    return chosen.format(name=name, fielder=fielder, pos=pos_label)


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

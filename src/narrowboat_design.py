# SPDX-License-Identifier: MIT
"""NB57 "Wren": an original 57 ft semi-traditional narrowboat, as data (stdlib only).

This module is the single source of truth for the design: the A3 drawing set
(narrowboat_blueprint) and the Blender model (narrowboat_common / _interior / _build) both
read it, so the drawing and the model cannot disagree. Values are millimetres.

Coordinates: b runs along the boat from the aft face of the rear cabin bulkhead, positive
towards the bow; y is positive to port; z is up from the underside of the baseplate.

The layout is traditional (saloon forward, bedroom aft) behind a semi-trad stern, with the
engine under the rear deck. Limits and rules it is designed to:
- Canal & River Trust waterway dimensions: 57 ft (17.37 m) over everything reaches the whole
  connected network (Ripon / Calder & Hebble locks); 6'10" (2.08 m) beam for Standedge and the
  Huddersfield Narrow; Standedge's cabin-corner profile table (STANDEDGE_PROFILE).
- Boat Safety Scheme: fixed ventilation V = 2200U + 650P + 440F + 550H mm², split equally high
  and low; LPG locker drained overboard from its lowest point through >= 19 mm ID, exiting
  above the deepest laden waterline, opening only to the outside; solid-fuel stove <= 4.5 kW;
  a CO alarm (BS EN 50291) is mandatory.
- BS 8511 solid-fuel stoves on boats: hearth >= 225 mm in front and >= 150 mm each side;
  combustibles 600 mm away or behind a protection panel (10 mm air gap, 25 mm calcium
  silicate, >= 45 mm to the stove, >= 200 mm above it); flue >= 100 mm, chimney removable.
- Builders' practice: 10/6/5/4 mm plate (base/hull/cabin/roof), 75 mm roof camber, 50x13 mm
  rubbing strakes, ~38 mm spray foam, 1220 x 1905 (4'0" x 6'3") double berth, 800 mm quadrant
  shower, 21" (533 mm) tall windows and 15" (381 mm) portholes.
"""

from __future__ import annotations

from dataclasses import dataclass

NAME = "Wren"
DESIGN_ID = "NB57"
TITLE = "57 FT SEMI-TRAD NARROWBOAT"

# --------------------------------------------------------------------------- lengths (b)

STERN_TIP = -1850  # aft end of the counter plating
STEM = 15300  # forward face of the stem at the sheer
FENDER_AFT, FENDER_FWD = 100, 120  # rope button fenders beyond the steel
FENDER_AFT_Z, FENDER_FWD_Z, FENDER_H = (
    850,
    1130,
    260,
)  # centre heights; bow one on the stem head
CABIN_LEN = 12850  # rear bulkhead (b = 0) to front bulkhead
SHELTER_AFT = -900  # semi-trad side plates run aft of the rear bulkhead to here
WELL_DECK_END = (
    14100  # aft wall of the gas locker; the well deck runs from the front bulkhead
)
LOCKER_END = 14800  # forward wall of the gas locker
STERN_POST = -1000  # where the swim closes on the stern tube
SWIM_START = 850  # the baseplate starts to narrow here, under the stern steps

HULL_LENGTH = STEM - STERN_TIP  # 17,150
LENGTH_OVER_FENDERS = HULL_LENGTH + FENDER_AFT + FENDER_FWD  # 17,370 = 57'0"
CRT_MAX_LENGTH = 17370  # Ripon 17.37 m; the Calder & Hebble's 57'6" locks are 17.53 m

# --------------------------------------------------------------------------- sections (z, y)

HALF_BEAM = 1027  # over the 6 mm side plating
STRAKE_H, STRAKE_PROUD = 50, 13  # 50 x 13 convex rubbing strakes
BEAM_OVER_STRAKES = 2 * (HALF_BEAM + STRAKE_PROUD)  # 2,080 = 6'10"
CRT_MAX_BEAM = 2080  # Standedge / Huddersfield Narrow
Z_STRAKES = (680, 830, 985)  # strake centrelines, between the waterline and the gunwale
GUNWALE_W = 100
Z_GUNWALE = 1080
CABIN_BASE_HALF = HALF_BEAM - GUNWALE_W  # cabin side's outer face at the gunwale
TUMBLEHOME = 50  # the cabin side leans in this much from gunwale to roof corner
Z_SIDE_TOP = 1990  # cabin side meets the roof's corner radius
CORNER_R = 45
Z_ROOF_EDGE = Z_SIDE_TOP + CORNER_R
CAMBER = 75
Z_ROOF_CROWN = Z_ROOF_EDGE + CAMBER  # 2,110
CABIN_TOP_HALF = CABIN_BASE_HALF - TUMBLEHOME

# Plate thicknesses.
T_BASE, T_HULL, T_CABIN, T_ROOF, T_BULKHEAD = 10, 6, 5, 4, 6

# Linings: spray foam between battens, then ply.
INSULATION_SIDE = 45  # 38 mm foam on 25 x 45 battens, side walls
INSULATION_ROOF = 45
PLY = 9
FLOOR_PLY = 18

# Hull lines: control points (distance, value) for monotone cubics. Each set is keyed from
# where it leaves the parallel body, so changing the length moves it as a whole.
BOW_LEN = 3000  # entrance: full beam this far aft of the stem
BOW_PLAN = (
    (0, HALF_BEAM),
    (800, 1005),
    (1600, 900),
    (2300, 640),
    (2750, 330),
    (3000, 40),
)
SHEER_LEN = 3200  # gunwale rises to the stem over this length
SHEER = ((0, Z_GUNWALE), (1000, 1100), (2000, 1170), (2700, 1250), (3200, 1330))
FOREFOOT_LEN = 650  # the bottom starts to rise this far aft of the stem
FOREFOOT = ((0, 0), (200, 40), (380, 160), (520, 420), (600, 800), (650, 1290))
SWIM_LEN = SWIM_START - STERN_POST  # 1,850 (6'1") of tapering bottom to the stern post
SWIM = ((0, HALF_BEAM), (500, 960), (1000, 760), (1450, 430), (1850, 50))
# Height at which the swim is full width again (distance from the swim start, z): low at the
# start, rising to the counter at the stern post; below it the side flares in over SWIM_FLARE.
SWIM_Z = ((0, 250), (800, 340), (1400, 470), (1850, 590))
SWIM_FLARE = 150
CHINE = 20  # small chamfer where the baseplate meets the sides
Z_COUNTER = 600  # underside of the counter, over the propeller

# Waterline, trimmed by the stern: deep enough aft to immerse an 18" propeller, high enough
# forward for the gas locker drain. Draught marks are at the stern post and the stem.
DRAUGHT_AFT, DRAUGHT_FWD = 620, 480
AIR_DRAUGHT_MAX = (
    1750  # CRT lowest bridges/tunnels 1.78 m; fixed fittings stay under this
)

# Standedge tunnel cabin profile: (height of the cabin corner above water, max cabin top
# width), both converted from the CRT table's feet and inches.
STANDEDGE_PROFILE = (
    (1473, 2032),  # 4'10" -> 6'8"
    (1524, 1981),  # 5'0" -> 6'6"
    (1549, 1930),  # 5'1" -> 6'4"
    (1575, 1880),  # 5'2" -> 6'2"
    (1600, 1829),  # 5'3" -> 6'0"
    (1626, 1727),  # 5'4" -> 5'8"
    (1651, 1676),  # 5'5" -> 5'6"
    (1702, 1575),  # 5'7" -> 5'2"
    (1727, 1524),  # 5'8" -> 5'0"
    (1753, 1422),  # 5'9" -> 4'8"
    (1778, 1372),  # 5'10" -> 4'6"
    (1803, 1321),  # 5'11" -> 4'4"
    (1829, 1270),  # 6'0" -> 4'2"
    (1854, 1219),  # 6'1" -> 4'0"
    (1880, 1219),  # 6'2" -> 4'0"
)

# --------------------------------------------------------------------------- interior

# Main cabin floor, built up from the top of the baseplate: (z0, z1, layer).
FLOOR_Z = 150
FLOOR_BUILDUP = (
    (10, 60, "concrete ballast slabs on bitumen"),
    (60, 72, "packers"),
    (72, 122, "50 x 50 bearers"),
    (122, 140, "18 ply floor"),
    (140, 150, "finish"),
)
LOBBY_FLOOR_Z = 420  # cupboard plinths over the swim beside the stern steps
LINING_IN = (
    HALF_BEAM - T_HULL - INSULATION_SIDE - PLY
)  # 967: inner face below the gunwale
LINING_AFT, LINING_FWD = 60, CABIN_LEN - 60  # end-lining faces inside the bulkheads
CEILING_CROWN = Z_ROOF_CROWN - T_ROOF - INSULATION_ROOF - PLY  # 2,052
HEADROOM = CEILING_CROWN - FLOOR_Z  # 1,902 (6'3")
PARTITION_T = 18


@dataclass(frozen=True)
class Zone:
    name: str
    b0: int
    b1: int
    floor_z: int
    floor: str  # finish slots, resolved to materials by the build
    walls: str
    ceiling: str


ZONES = (
    Zone("Lobby", LINING_AFT, 850, LOBBY_FLOOR_Z, "Oak", "Ash", "CreamTG"),
    Zone("Bedroom", 850, 3450, FLOOR_Z, "Carpet", "Ash", "CreamTG"),
    Zone("Bathroom", 3450, 5350, FLOOR_Z, "Vinyl", "SageTG", "CreamTG"),
    Zone("Galley", 5350, 8100, FLOOR_Z, "Oak", "Ash", "CreamTG"),
    Zone("Saloon", 8100, LINING_FWD, FLOOR_Z, "Oak", "Ash", "CreamTG"),
)
ZONE = {z.name: z for z in ZONES}


@dataclass(frozen=True)
class Partition:
    """A full-height ply partition at b with a doorway (y0, y1) and a hinged door."""

    name: str
    b: int
    door_y: tuple[int, int]
    hinge: int  # side of the doorway the door hangs on (+1 port)
    opens: int  # +1 swings forward, -1 aft


PARTITIONS = (
    # The bedroom corridor runs to port, so the aft bathroom door is to port; the forward
    # door is on the centreline, in line with the galley aisle.
    Partition("BathroomAft", 3450, (310, 927), -1, 1),
    Partition("BathroomFwd", 5350, (-310, 310), 1, -1),
)
DOOR_H = 1850  # above the floor


@dataclass(frozen=True)
class Fixture:
    """One piece of furniture or equipment.

    Against a lining (`side` = +1 port / -1 starboard) it runs `depth` inboard from the
    lining's inner face; free-standing pieces (`side` = 0) give `y` directly. z is absolute.
    `host` names the run an appliance is set into; `opts` holds builder-specific options.
    """

    name: str
    kind: str
    zone: str
    b0: int
    b1: int
    z0: int
    z1: int
    side: int = 0
    depth: int = 0
    y: tuple[int, int] = (0, 0)
    host: str = ""
    opts: tuple = ()

    def footprint(self) -> tuple[int, int, int, int]:
        """(b0, b1, y0, y1) in plan."""
        if self.side:
            outer = self.side * LINING_IN
            inner = self.side * (LINING_IN - self.depth)
            return (self.b0, self.b1, min(outer, inner), max(outer, inner))
        return (self.b0, self.b1, *self.y)

    def opt(self, key: str, default=None):
        for k, v in self.opts:
            if k == key:
                return v
        return default


F = FLOOR_Z
WORKTOP_Z = (
    F + 900
)  # worktop surface, just under the gunwale ledge so tops run to the hull
FULL = 1950  # tall units stop under the ceiling where it meets the cabin-side lining

FIXTURES = (
    # Lobby: stern steps from the floor up to the doorway sill, cupboards over the swim.
    Fixture(
        "SternSteps",
        "steps",
        "Lobby",
        60,
        850,
        F,
        Z_GUNWALE + 30,
        y=(-300, 300),
        opts=(("risers", 4), ("rise_to", -1)),
    ),
    Fixture(
        "CalorifierCupboard",
        "cupboard",
        "Lobby",
        60,
        850,
        LOBBY_FLOOR_Z,
        FULL,
        side=1,
        depth=630,
        opts=(("contents", "55 L calorifier"),),
    ),
    Fixture(
        "ElectricsCupboard",
        "cupboard",
        "Lobby",
        60,
        850,
        LOBBY_FLOOR_Z,
        FULL,
        side=-1,
        depth=630,
        opts=(("contents", "consumer unit, inverter-charger"),),
    ),
    # Bedroom: wardrobe and a fixed 4'0" x 6'3" side bed to starboard, corridor to port.
    Fixture("Wardrobe", "wardrobe", "Bedroom", 860, 1460, F, FULL, side=-1, depth=600),
    Fixture(
        "Bed",
        "bed",
        "Bedroom",
        1480,
        3430,
        F,
        F + 520,
        side=-1,
        depth=1220,
        opts=(("mattress", (1905, 1220)), ("head", 1)),
    ),
    # Bathroom: walk-through, fittings to starboard, towel radiator to port.
    Fixture(
        "Shower",
        "shower",
        "Bathroom",
        3470,
        4270,
        F,
        F + 1800,
        side=-1,
        depth=800,
        opts=(("corner", "aft"),),
    ),
    Fixture("Vanity", "vanity", "Bathroom", 4290, 4790, F, F + 850, side=-1, depth=400),
    Fixture(
        "Toilet",
        "toilet",
        "Bathroom",
        4860,
        5254,
        F,
        F + 534,
        side=-1,
        depth=580,
        opts=(("type", "cassette, 394 x 580"),),
    ),
    Fixture(
        "TowelRadiator",
        "towel_rail",
        "Bathroom",
        3700,
        4200,
        F + 300,
        F + 900,
        side=1,
        depth=80,
    ),
    # Galley: worktops both sides, 600 deep, leaving a 734 mm aisle.
    Fixture(
        "GalleyPort", "base_run", "Galley", 5370, 8100, F, WORKTOP_Z, side=1, depth=600
    ),
    Fixture(
        "GalleyStarboard",
        "base_run",
        "Galley",
        5870,
        8100,
        F,
        WORKTOP_Z,
        side=-1,
        depth=600,
    ),
    Fixture("Larder", "tall_unit", "Galley", 5370, 5870, F, FULL, side=-1, depth=600),
    Fixture(
        "Fridge",
        "fridge",
        "Galley",
        5390,
        5940,
        F,
        WORKTOP_Z - 40,
        side=1,
        depth=560,
        host="GalleyPort",
    ),
    Fixture(
        "Sink",
        "sink",
        "Galley",
        6420,
        7300,
        WORKTOP_Z - 180,
        WORKTOP_Z,
        side=1,
        depth=480,
        host="GalleyPort",
        opts=(("bowl", (6420, 6820)),),
    ),
    Fixture(
        "Cooker",
        "cooker",
        "Galley",
        5900,
        6400,
        F,
        WORKTOP_Z,
        side=-1,
        depth=560,
        host="GalleyStarboard",
        opts=(("burners", 4),),
    ),
    Fixture(
        "WallUnitsPortAft",
        "wall_units",
        "Galley",
        5370,
        6340,
        F + 1450,
        FULL,
        side=1,
        depth=320,
    ),
    Fixture(
        "WallUnitsPortFwd",
        "wall_units",
        "Galley",
        7260,
        8050,
        F + 1450,
        FULL,
        side=1,
        depth=320,
    ),
    Fixture(
        "WallUnitStarboard",
        "wall_units",
        "Galley",
        6450,
        6990,
        F + 1450,
        FULL,
        side=-1,
        depth=320,
    ),
    # Saloon: stove on its hearth to port, Pullman dinette under the side hatch to starboard.
    Fixture(
        "Hearth",
        "hearth",
        "Saloon",
        8250,
        8950,
        F,
        F + 50,
        side=1,
        depth=LINING_IN - 250,
    ),
    Fixture(
        "Stove",
        "stove",
        "Saloon",
        8405,
        8795,
        F + 50,
        F + 50 + 546,
        y=(475, 845),
        opts=(("kw", 4.5), ("flue_d", 125)),
    ),
    Fixture(
        "HeatShieldWall",
        "heat_shield",
        "Saloon",
        8250,
        8950,
        F + 50,
        F + 50 + 546 + 250,
        side=1,
        depth=35,
    ),
    Fixture(
        "HeatShieldGalley",
        "heat_shield",
        "Saloon",
        8100,
        8135,
        F,
        WORKTOP_Z,
        side=1,
        depth=600,
    ),
    Fixture(
        "Dinette",
        "dinette",
        "Saloon",
        8120,
        9800,
        F,
        F + 900,
        side=-1,
        depth=620,
        opts=(
            ("benches", ((8120, 8570), (9350, 9800))),
            ("table", (8570, 9350)),
            ("seat_z", F + 450),
        ),
    ),
    Fixture(
        "Sofa",
        "sofa",
        "Saloon",
        9700,
        11300,
        F,
        F + 850,
        side=1,
        depth=750,
        opts=(("seat_z", F + 450),),
    ),
    Fixture(
        "Armchair",
        "armchair",
        "Saloon",
        11330,
        12080,
        F,
        F + 900,
        side=-1,
        depth=750,
        opts=(("seat_z", F + 450),),
    ),
    Fixture(
        "Shelves", "shelves", "Saloon", 11400, 12100, F, F + 900, side=1, depth=400
    ),
    Fixture(
        "BowSteps",
        "steps",
        "Saloon",
        12130,
        LINING_FWD,
        F,
        950,
        y=(-300, 300),
        opts=(("risers", 4), ("rise_to", 1)),
    ),
    Fixture(
        "CoatLocker",
        "cupboard",
        "Saloon",
        12130,
        LINING_FWD,
        F,
        FULL,
        side=-1,
        depth=630,
    ),
    Fixture(
        "LogLocker",
        "cupboard",
        "Saloon",
        12130,
        LINING_FWD,
        F,
        F + 900,
        side=1,
        depth=630,
    ),
)
FIXTURE = {f.name: f for f in FIXTURES}
# Pieces whose footprints may not overlap one another (appliances sit inside their runs and
# the stove and its shields sit on or against the hearth, so they are left out).
CARCASS_KINDS = frozenset(
    {
        "steps",
        "cupboard",
        "wardrobe",
        "bed",
        "shower",
        "vanity",
        "toilet",
        "base_run",
        "tall_unit",
        "hearth",
        "dinette",
        "sofa",
        "armchair",
        "shelves",
    }
)

# Safety equipment for the plan: (symbol, b, y). FE = 5A/34B extinguisher, FB = fire blanket,
# CO = carbon monoxide alarm (BS EN 50291), SA = smoke alarm.
SAFETY = (
    ("FE", 960, 800),
    ("FE", 7950, 330),
    ("FE", 12020, 150),
    ("FB", 6460, -400),
    ("CO", 2100, 600),
    ("CO", 9900, 0),
    ("SA", 7600, 0),
)

# --------------------------------------------------------------------------- openings

WINDOW_H, WINDOW_R = 533, 70  # 21" tall apertures with 70 mm corner radius
WINDOW_SILL = 1210
WINDOW_ZC = WINDOW_SILL + WINDOW_H // 2  # 1,476 (rounded down): one band both sides
PORTHOLE_D = 381  # 15"
FRAME_FLANGE, RIM_W = 40, 50  # window frame flange and porthole rim beyond the aperture


@dataclass(frozen=True)
class Opening:
    """A cabin-side aperture centred at b on `side` (+1 port)."""

    name: str
    side: int
    b: int
    w: int
    h: int
    r: int
    kind: str = "window"  # window | porthole
    zc: int = WINDOW_ZC

    @property
    def span(self) -> tuple[int, int]:
        """b extent of the paint it interrupts: the aperture plus its flange or rim."""
        rim = RIM_W if self.kind == "porthole" else FRAME_FLANGE
        return (self.b - self.w // 2 - rim, self.b + self.w // 2 + rim)


def _w(name: str, side: int, b: int, width: int) -> Opening:
    return Opening(name, side, b, width, WINDOW_H, WINDOW_R)


def _p(name: str, side: int, b: int) -> Opening:
    return Opening(name, side, b, PORTHOLE_D, PORTHOLE_D, PORTHOLE_D // 2, "porthole")


OPENINGS = (
    _w("PortBedroom", 1, 2250, 762),
    _p("PortBathroom", 1, 4450),
    _w("PortGalley", 1, 6800, 762),
    _w("PortSaloon", 1, 10150, 914),
    _p("PortSaloonFwd", 1, 11550),
    _w("StarboardBedroom", -1, 2450, 914),
    _p("StarboardBathroom", -1, 4540),
    _w("StarboardGalley", -1, 7450, 762),
    _w("StarboardSaloon", -1, 10700, 914),
    _p("StarboardSaloonFwd", -1, 11750),
)


@dataclass(frozen=True)
class SideHatch:
    side: int
    b: int  # centre
    clear: int  # clear opening between the leaves' frames
    z0: int  # bottom of the leaves
    leaf_h: int

    @property
    def span(self) -> tuple[int, int]:
        return (self.b - self.clear // 2, self.b + self.clear // 2)


SIDE_HATCH = SideHatch(-1, 8960, 700, Z_GUNWALE + 120, 620)


@dataclass(frozen=True)
class Doors:
    """Double doors on a cabin bulkhead: leaves meet on the centreline."""

    end: str  # Stern | Bow
    b: int
    half: int  # each leaf spans 5..half from the centreline
    z0: int
    z1: int


DOORS = (
    Doors("Stern", 0, 310, Z_GUNWALE + 30, 1930),
    # The bow doors stand on a 150 mm upstand above the well-deck sole to keep water out.
    Doors("Bow", CABIN_LEN, 310, 950, 1930),
)

# Cabin coachline panel and where the cabin-side rose sprays go: (design, side, b hint).
PANEL_B = (250, CABIN_LEN - 250)
PANEL_Z = (1115, 1935)
SPRAYS = (
    ("CabinAft", 1, 1050),
    ("CabinCentre", 1, 8400),
    ("CabinFore", 1, 12150),
    ("CabinAft", -1, 1100),
    ("CabinCentre", -1, 5800),
    ("CabinFore", -1, 9750),
)

# --------------------------------------------------------------------------- roof and decks

HANDRAIL_B = (350, CABIN_LEN - 350)
HANDRAIL_INSET = 90  # handrail centre inboard of the roof corner
HANDRAIL_H, HANDRAIL_D = 75, 26
VENT_BS = (2000, 4000, 6150, 10400)  # mushroom vents on the centreline
VENT_D, VENT_H = 150, 86
CHIMNEY = (8600, 660)  # (b, y) above the stove's flue
CHIMNEY_D, CHIMNEY_H, COLLAR_D = 130, 700, 180
REAR_HATCH = (20, 720, 350)  # sliding hatch over the stern doors: b0, b1, half-width
PLANK_Y = -400  # pole and plank along the starboard roof (the chimney is to port)
PLANK_BLOCKS = (2600, 6600, 10600)
PLANK_B = (2100, 11100)
PLANK_W, PLANK_T, BLOCK_H = 260, 32, 40  # plank on blocks that sit on the roof
BLOCK_Y = (-560, -140)  # blocks carry the plank and, inboard of it, the pole
POLE_Y, POLE_D = -195, 44
POLE_B, POLE_LEN = 4200, 4200

DECK_Z = Z_GUNWALE  # semi-trad rear deck and counter deck
ENGINE_BOARDS = (-880, -20)  # lift-out boards over the engine
WEED_HATCH = (-1330, -1030, 180)  # b0, b1, half-width: over the propeller
WELL_SOLE_Z = 800  # self-draining well deck, well above the forward waterline
SEAT_LOCKERS = (-880, -120, 380, 420)  # b0, b1, depth from the shelter side, height
T_STUDS = ((STEM - 350, 0), (-1500, 600), (-1500, -600))
DIESEL_FILLER = (-1620, -380)  # on the counter deck over the tank, clear of the stock
WATER_FILLER = (13650, -650)  # on the well-deck sole over the tank
# Swan neck: the rudder stock stands STOCK_TOP above the deck, the neck rises NECK_RISE more
# and reaches NECK_REACH forward to the tiller bar, which runs TILLER_BAR forward of it.
STOCK_TOP, NECK_RISE, NECK_REACH, TILLER_BAR = 320, 300, 340, 850

# --------------------------------------------------------------------------- machinery

# ~43 hp diesel on bearers; they rise aft where the swim narrows beneath it.
ENGINE = (-700, -100, -280, 280, 260, 820)
GEARBOX = (-900, -700, -120, 120, 260, 560)  # narrow, where the swim closes in
PROP_B, PROP_Z, PROP_D, PROP_BLADES = -1180, 300, 457, 3  # 18" three-blade
RUDDER = (-1830, -1380, 60, 560)  # balanced blade: b0, b1, z0, z1
RUDDER_STOCK_B = -1490  # about a quarter of the chord behind the leading edge
STOCK_D = 50
SKEG_END = -1560  # skeg runs from the stern post to the rudder's bottom bearing
BATTERIES = (-600, -150, -880, -480, 500, 750)  # vented box, engine bay to starboard
# Integral tanks: b0, b1, z0, z1 between the plating.
FUEL_TANK = (-1800, -1350, 660, 1040)  # in the counter, rudder tube welded through
WATER_TANK = (12880, 13780, 320, 720)  # under the well deck
GAS_LOCKER = (WELL_DECK_END, LOCKER_END)  # b; its sole sits LOCKER_DEPTH under the lid
LOCKER_DEPTH = 660  # 13 kg propane cylinder 580 tall + regulator and pigtails
LOCKER_COAMING = 80  # the lid stands proud of the foredeck by this much
GAS_CYLINDER = (315, 580)  # 13 kg propane, diameter x height
GAS_CYLINDERS = 2
LOCKER_DRAIN_ID = 19  # BSS minimum internal diameter

# --------------------------------------------------------------------------- ventilation


@dataclass(frozen=True)
class Vent:
    name: str
    level: str  # high | low
    # Free area in mm²: a stated design assumption, to confirm against the fitted product.
    free_area: int
    count: int


PERSONS = 4  # double berth plus the dinette and sofa as occasional berths
UNFLUED_KW = 7.0  # LPG cooker: four-burner hob plus oven at full rate
FLUED_KW = 0.0  # no open- or closed-flued (non room-sealed) appliances
STOVE_KW = 4.5
VENTS = (
    Vent("Roof mushroom vent", "high", 3000, len(VENT_BS)),
    Vent("Door lower-panel louvre", "low", 2500, 4),
    Vent("Cabin-side vent at gunwale", "low", 2500, 2),
)


def ventilation_required() -> float:
    """BSS fixed ventilation, mm²: 2200U + 650P + 440F + 550H."""
    return 2200 * UNFLUED_KW + 650 * PERSONS + 440 * FLUED_KW + 550 * STOVE_KW


def ventilation_provided(level: str) -> int:
    return sum(v.free_area * v.count for v in VENTS if v.level == level)


# --------------------------------------------------------------------------- weights (kg)

STEEL_DENSITY = 7850  # kg/m³
FRAMING_ALLOWANCE = 0.18  # frames, knees, roof bearers, stiffeners as a share of plate
WATER_DENSITY = 1000  # fresh water, kg/m³
DIESEL_DENSITY = 840
FITOUT_KG = 2600  # linings, insulation, floors, joinery, upholstery, wiring, plumbing
ITEMS_KG = (
    ("Engine and gearbox", 290),
    ("Stove", 70),
    ("Cooker, fridge, calorifier", 110),
    ("Batteries (4 x 110 Ah + starter)", 150),
    ("LPG, 2 x 13 kg full", 56),
    ("Crew (4)", 320),
    ("Stores and gear", 200),
)

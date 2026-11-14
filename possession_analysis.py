import cv2 as cv
import numpy as np
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from ultralytics import YOLO

# ============================================================
# 1. CONFIGURATION
# ============================================================

VIDEO_FILE = "BAR-ATH_tactical-001.mp4"
ATD_FILE = "BAR-ATH_tactical_atd.txt"
XML_FILE = "BAR-ATH_tactical_atd.xml"
HOMOGRAPHY_FILE = "BAR-ATH_tactical_homography.csv"
OUTPUT_CSV = "possession_analysis_final.csv"

START_TIME = 0.0
ATD_FPS = 25.0

POSSESSION_DISTANCE = 0.035
BOUNDARY_MARGIN = 0.01

HISTORY_LENGTH = 10
MIN_VALID_VOTES = 4
POSSESSION_MAJORITY = 0.65
MAX_HOLD_FRAMES = 5

YOLO_CONFIDENCE = 0.10
YOLO_IMAGE_SIZE = 1536

# ============================================================
# 2. INITIALISE YOLO AND VIDEO
# ============================================================

print("Loading YOLO model...")

model = YOLO("yolo26n.pt")
capture = cv.VideoCapture(VIDEO_FILE)

if not capture.isOpened():
    raise SystemExit("ERROR: Could not open video file.")

video_fps = capture.get(cv.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 25.0

video_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
video_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

capture.set(cv.CAP_PROP_POS_MSEC, START_TIME * 1000)

print("Video loaded.")
print("FPS:", video_fps)
print("Resolution:", video_width, "x", video_height)

# ============================================================
# 3. LOAD TEAM INFORMATION FROM XML
# ============================================================

print("\nReading ATD XML metadata...")

tree = ET.parse(XML_FILE)
root = tree.getroot()

team_names = {}
player_teams = {}

for team in root.iter("Team"):
    team_id = team.attrib.get("id")
    team_name = (
        team.attrib.get("name")
        or team.attrib.get("shortName")
        or team.attrib.get("displayName")
    )

    if team_id and team_name:
        team_names[team_id] = team_name

for player in root.iter("Player"):
    player_id = player.attrib.get("id")
    team_id = player.attrib.get("teamId")

    if not player_id or not team_id:
        continue

    if team_id == "MET_user_team_o4Hossvb8M":
        player_teams[int(player_id)] = "Barcelona"

    elif team_id == "MET_user_team_x7C4FFX2HF":
        player_teams[int(player_id)] = "Athletic Club"

    elif team_id in team_names:
        name = team_names[team_id].lower()

        if "barcelona" in name:
            player_teams[int(player_id)] = "Barcelona"

        elif "atletic" in name or "athletic" in name:
            player_teams[int(player_id)] = "Athletic Club"

barcelona_count = sum(
    team == "Barcelona" for team in player_teams.values()
)

athletic_count = sum(
    team == "Athletic Club" for team in player_teams.values()
)

print("Team information loaded.")
print("Barcelona:", barcelona_count)
print("Athletic Club:", athletic_count)

if len(player_teams) != 46:
    print(
        "WARNING: Expected 46 player assignments, found",
        len(player_teams)
    )

# ============================================================
# 4. LOAD HOMOGRAPHY FILE
# ============================================================

homography_available = True

try:
    with open(HOMOGRAPHY_FILE, "r") as f:
        homography_header = f.readline().strip()

    print("\nHomography file found.")

except FileNotFoundError:
    homography_available = False
    print("\nWARNING: Homography file not found.")

# ============================================================
# 5. JERSEY COLOUR AND ROLE CLASSIFICATION
# ============================================================

team_history = defaultdict(lambda: deque(maxlen=20))
role_history = defaultdict(lambda: deque(maxlen=12))


def get_jersey_features(frame, box):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    jx1 = max(0, x1 + int(width * 0.20))
    jx2 = min(frame.shape[1], x2 - int(width * 0.20))

    jy1 = max(0, y1 + int(height * 0.20))
    jy2 = min(frame.shape[0], y1 + int(height * 0.48))

    if jx2 <= jx1 or jy2 <= jy1:
        return None

    crop = frame[jy1:jy2, jx1:jx2]

    if crop.size == 0:
        return None

    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    red_mask = (
        ((h <= 14) | (h >= 168))
        & (s >= 45)
        & (v >= 30)
    )

    blue_mask = (
        (h >= 85)
        & (h <= 145)
        & (s >= 30)
        & (v >= 20)
    )

    white_mask = (
        (s <= 75)
        & (v >= 125)
    )

    total = h.size

    if total == 0:
        return None

    red_ratio = np.count_nonzero(red_mask) / total
    blue_ratio = np.count_nonzero(blue_mask) / total
    white_ratio = np.count_nonzero(white_mask) / total

    coloured_ratio = np.count_nonzero(
        (s >= 55) & (v >= 35)
    ) / total

    return (
        red_ratio,
        blue_ratio,
        white_ratio,
        coloured_ratio
    )


# ============================================================
# 6. GOALKEEPER AND ASSISTANT REFEREE FEATURES
# ============================================================

def get_role_features(frame, box):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    jx1 = max(0, x1 + int(width * 0.20))
    jx2 = min(frame.shape[1], x2 - int(width * 0.20))

    jy1 = max(0, y1 + int(height * 0.20))
    jy2 = min(frame.shape[0], y1 + int(height * 0.55))

    if jx2 <= jx1 or jy2 <= jy1:
        return None

    crop = frame[jy1:jy2, jx1:jx2]

    if crop.size == 0:
        return None

    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    total = h.size

    if total == 0:
        return None

    # Green / mint goalkeeper jersey.
    green_mask = (
        (h >= 55)
        & (h <= 105)
        & (s >= 35)
        & (v >= 65)
    )

    # Fluorescent yellow-green assistant referee jersey.
    fluorescent_mask = (
        (h >= 22)
        & (h <= 58)
        & (s >= 85)
        & (v >= 130)
    )

    return {
        "green_ratio": float(
            np.count_nonzero(green_mask) / total
        ),
        "fluorescent_ratio": float(
            np.count_nonzero(fluorescent_mask) / total
        )
    }


def classify_role_frame(role_features, box, frame_shape):

    if role_features is None:
        return "Unknown"

    green_ratio = role_features["green_ratio"]
    fluorescent_ratio = role_features["fluorescent_ratio"]

    x1, y1, x2, y2 = box

    frame_h, frame_w = frame_shape[:2]

    center_x = ((x1 + x2) / 2) / max(frame_w, 1)
    center_y = ((y1 + y2) / 2) / max(frame_h, 1)

    near_left_goal_area = (
        center_x <= 0.38
        and 0.15 <= center_y <= 0.82
    )

    near_right_goal_area = (
        center_x >= 0.62
        and 0.15 <= center_y <= 0.82
    )

    near_goal_area = (
        near_left_goal_area or near_right_goal_area
    )

    near_touchline = (
        center_y <= 0.18 or center_y >= 0.78
    )

    if fluorescent_ratio >= 0.18 and near_touchline:
        return "Assistant Referee?"

    if green_ratio >= 0.16 and near_goal_area:
        return "Goalkeeper?"

    return "Unknown"


def get_stable_role(track_id, frame_role):

    if track_id is None:
        return "Unknown"

    if frame_role != "Unknown":
        role_history[track_id].append(frame_role)

    votes = list(role_history[track_id])

    if len(votes) < 3:
        return "Unknown"

    for candidate in (
        "Goalkeeper?",
        "Assistant Referee?"
    ):
        if votes.count(candidate) >= 3:
            return candidate

    return "Unknown"


# ============================================================
# 7. TEAM CLASSIFICATION AND TEMPORAL VOTING
# ============================================================

def classify_frame(features):

    if features is None:
        return "Unknown"

    red, blue, white, coloured = features

    if red >= 0.10 and blue >= 0.025:
        return "Barcelona"

    if white >= 0.48 and coloured <= 0.30:
        return "Athletic Club"

    return "Unknown"


def get_stable_team(track_id, frame_team):

    if track_id is None:
        return "Unknown"

    if frame_team != "Unknown":
        team_history[track_id].append(frame_team)

    votes = list(team_history[track_id])

    if len(votes) < 5:
        return "Unknown"

    barcelona_votes = votes.count("Barcelona")
    athletic_votes = votes.count("Athletic Club")

    total_votes = len(votes)

    if (
        barcelona_votes >= 5
        and barcelona_votes / total_votes >= 0.70
    ):
        return "Barcelona"

    if (
        athletic_votes >= 5
        and athletic_votes / total_votes >= 0.70
    ):
        return "Athletic Club"

    return "Unknown"


# ============================================================
# 8. READ ATD TRACKING DATA
# ============================================================

def parse_atd_line(line):

    parts = line.strip().split(":")

    if len(parts) != 3:
        return None

    try:
        frame_id = int(parts[0])
    except ValueError:
        return None

    player_entries = parts[1].split(";")

    if len(player_entries) != 46:
        return None

    players = []

    for index, entry in enumerate(player_entries):

        values = entry.split(",")

        if len(values) != 3:
            continue

        try:
            x, y, speed = map(float, values)
        except ValueError:
            continue

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        player_id = index + 1
        team = player_teams.get(player_id, "Unknown")

        players.append({
            "id": player_id,
            "team": team,
            "x": x,
            "y": y,
            "speed": speed
        })

    ball_values = parts[2].split(",")

    ball = None

    if len(ball_values) == 3:
        try:
            ball_x = float(ball_values[0])
            ball_y = float(ball_values[1])
            ball_speed = float(ball_values[2])

            if np.isfinite(ball_x) and np.isfinite(ball_y):
                ball = {
                    "x": ball_x,
                    "y": ball_y,
                    "speed": ball_speed
                }

        except ValueError:
            pass

    return {
        "frame": frame_id,
        "players": players,
        "ball": ball
    }


# ============================================================
# 9. TEMPORAL POSSESSION CLASSIFICATION
# ============================================================

possession_history = deque(maxlen=HISTORY_LENGTH)

last_stable_possession = "X"
frames_since_stable = 0


def classify_dead_ball(ball):

    if ball is None:
        return False

    x = ball["x"]
    y = ball["y"]

    outside = (
        x < -BOUNDARY_MARGIN
        or x > 1.0 + BOUNDARY_MARGIN
        or y < -BOUNDARY_MARGIN
        or y > 1.0 + BOUNDARY_MARGIN
    )

    return outside


def estimate_raw_possession(players, ball):

    if ball is None:
        return "X", "No ball coordinates", None

    if classify_dead_ball(ball):
        return "D", "Ball outside pitch", None

    ball_x = ball["x"]
    ball_y = ball["y"]

    nearest_player = None
    minimum_distance = float("inf")

    for player in players:

        if player["team"] not in (
            "Barcelona",
            "Athletic Club"
        ):
            continue

        distance = np.sqrt(
            (ball_x - player["x"]) ** 2
            + (ball_y - player["y"]) ** 2
        )

        if distance < minimum_distance:
            minimum_distance = distance
            nearest_player = player

    if nearest_player is None:
        return "X", "No classified player", None

    if minimum_distance > POSSESSION_DISTANCE:
        return "X", "Ball too far from players", minimum_distance

    if nearest_player["team"] == "Barcelona":
        return "H", "Barcelona", minimum_distance

    if nearest_player["team"] == "Athletic Club":
        return "A", "Athletic Club", minimum_distance

    return "X", "Unknown", minimum_distance


def smooth_possession(raw):

    global last_stable_possession
    global frames_since_stable

    if raw == "D":
        possession_history.clear()
        last_stable_possession = "X"
        frames_since_stable = 0
        return "D"

    possession_history.append(raw)

    valid_votes = [
        item for item in possession_history
        if item in ("H", "A")
    ]

    if len(valid_votes) > 0:

        h_votes = valid_votes.count("H")
        a_votes = valid_votes.count("A")
        total = len(valid_votes)

        if (
            total >= MIN_VALID_VOTES
            and h_votes / total >= POSSESSION_MAJORITY
        ):
            last_stable_possession = "H"
            frames_since_stable = 0
            return "H"

        if (
            total >= MIN_VALID_VOTES
            and a_votes / total >= POSSESSION_MAJORITY
        ):
            last_stable_possession = "A"
            frames_since_stable = 0
            return "A"

    if raw == "X" and last_stable_possession in ("H", "A"):

        frames_since_stable += 1

        if frames_since_stable <= MAX_HOLD_FRAMES:
            return last_stable_possession

    if raw in ("H", "A"):
        frames_since_stable += 1

    return "X"


# ============================================================
# 10. OPEN ATD FILE AT THE CORRECT START FRAME
# ============================================================

target_atd_frame = int(round(START_TIME * ATD_FPS))

print("\nTarget timestamp:", START_TIME)
print("Target ATD frame:", target_atd_frame)

atd_file = open(ATD_FILE, "r")

for _ in range(target_atd_frame - 1):

    skipped_line = atd_file.readline()

    if not skipped_line:
        raise SystemExit(
            "ERROR: ATD file ended before the target frame."
        )

current_atd_frame = target_atd_frame

print("ATD file positioned successfully.")

# ============================================================
# 11. OUTPUT STATISTICS
# ============================================================

counts = {
    "H": 0,
    "A": 0,
    "D": 0,
    "X": 0
}

# ============================================================
# 12. MAIN ANALYSIS LOOP
# ============================================================

print("\nStarting possession analysis...\n")

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "video_frame",
    "timestamp_seconds",
    "atd_frame",
    "raw_possession",
    "final_possession",
    "ball_x",
    "ball_y",
    "ball_speed",
    "nearest_team",
    "nearest_player_id",
    "distance_pitch_coordinates",
    "visible_yolo_players",
    "visible_team_classified_players",
    "visible_goalkeeper_candidates",
    "visible_assistant_referee_candidates"
])

video_frame = 0

try:

    while True:

        success, frame = capture.read()

        if not success:
            break

        video_frame += 1

        timestamp = (
            START_TIME
            + (video_frame - 1) / video_fps
        )

        # ----------------------------------------------------
        # A. GET MATCHING ATD FRAME
        # ----------------------------------------------------

        required_atd_frame = int(
            round(timestamp * ATD_FPS)
        )

        atd_data = None

        while current_atd_frame < required_atd_frame:

            line = atd_file.readline()

            if not line:
                break

            current_atd_frame += 1

        if current_atd_frame == required_atd_frame:

            line = atd_file.readline()

            if line:
                atd_data = parse_atd_line(line)
                current_atd_frame += 1

        # ----------------------------------------------------
        # B. YOLO + BYTETRACK
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False
        )

        visible_players = []
        visible_roles = []
        visible_ball = None

        for box in results[0].boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            track_id = (
                int(box.id[0].item())
                if box.id is not None
                else None
            )

            # PLAYER DETECTION
            if class_id == 0:

                features = get_jersey_features(
                    frame,
                    (x1, y1, x2, y2)
                )

                frame_team = classify_frame(features)

                team = get_stable_team(
                    track_id,
                    frame_team
                )

                role_features = get_role_features(
                    frame,
                    (x1, y1, x2, y2)
                )

                frame_role = classify_role_frame(
                    role_features,
                    (x1, y1, x2, y2),
                    frame.shape
                )

                role = get_stable_role(
                    track_id,
                    frame_role
                )

                visible_roles.append(role)

                color = (200, 200, 200)

                if team == "Barcelona":
                    color = (255, 0, 0)

                elif team == "Athletic Club":
                    color = (0, 0, 255)

                visible_players.append(team)

                cv.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    1
                )

                foot_x = (x1 + x2) // 2
                foot_y = y2

                cv.circle(
                    frame,
                    (foot_x, foot_y),
                    4,
                    (0, 255, 0),
                    -1
                )

                display_class = team

                if role == "Goalkeeper?":
                    display_class = "GK candidate"
                    color = (0, 255, 0)

                elif role == "Assistant Referee?":
                    display_class = "Assistant referee?"
                    color = (0, 255, 255)

                label = (
                    f"ID {track_id} - {display_class}"
                    if track_id is not None
                    else display_class
                )

                cv.putText(
                    frame,
                    label,
                    (x1, max(y1 - 4, 12)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1
                )

            # BALL DETECTION
            elif class_id == 32:

                width = x2 - x1
                height = y2 - y1

                if width <= 35 and height <= 35:

                    visible_ball = (
                        (x1 + x2) // 2,
                        (y1 + y2) // 2
                    )

                    cv.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        2
                    )

        # ----------------------------------------------------
        # C. ATD POSSESSION ESTIMATION
        # ----------------------------------------------------

        raw_possession = "X"
        final_possession = "X"
        nearest_team = "Unknown"
        nearest_distance = None
        nearest_player_id = None

        ball_x = None
        ball_y = None
        ball_speed = None

        if atd_data is not None:

            atd_players = atd_data["players"]
            atd_ball = atd_data["ball"]

            raw_possession, nearest_team, nearest_distance = (
                estimate_raw_possession(
                    atd_players,
                    atd_ball
                )
            )

            final_possession = smooth_possession(
                raw_possession
            )

            if atd_ball is not None:
                ball_x = atd_ball["x"]
                ball_y = atd_ball["y"]
                ball_speed = atd_ball["speed"]

            if nearest_distance is not None:

                eligible_players = [
                    p for p in atd_players
                    if p["team"] in (
                        "Barcelona",
                        "Athletic Club"
                    )
                ]

                if eligible_players and atd_ball is not None:

                    nearest_player = min(
                        eligible_players,
                        key=lambda p: np.sqrt(
                            (atd_ball["x"] - p["x"]) ** 2
                            + (atd_ball["y"] - p["y"]) ** 2
                        )
                    )

                    nearest_player_id = nearest_player["id"]

        else:

            possession_history.clear()
            last_stable_possession = "X"
            frames_since_stable = 0

        # ----------------------------------------------------
        # D. COUNT POSSESSION STATES
        # ----------------------------------------------------

        counts[final_possession] += 1

        # ----------------------------------------------------
        # E. DISPLAY POSSESSION
        # ----------------------------------------------------

        labels = {
            "H": "BARCELONA",
            "A": "ATHLETIC CLUB",
            "D": "DEAD BALL",
            "X": "UNKNOWN"
        }

        colours = {
            "H": (255, 0, 0),
            "A": (0, 0, 255),
            "D": (0, 255, 255),
            "X": (180, 180, 180)
        }

        display_text = labels[final_possession]
        display_color = colours[final_possession]

        cv.rectangle(
            frame,
            (10, 10),
            (440, 150),
            (0, 0, 0),
            -1
        )

        cv.putText(
            frame,
            f"POSSESSION: {display_text}",
            (20, 45),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            display_color,
            2
        )

        cv.putText(
            frame,
            f"Raw label: {raw_possession}",
            (20, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv.putText(
            frame,
            f"ATD frame: {required_atd_frame}",
            (20, 110),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        if nearest_distance is not None:
            distance_text = (
                f"Player distance: {nearest_distance:.4f}"
            )
        else:
            distance_text = "Player distance: unavailable"

        cv.putText(
            frame,
            distance_text,
            (20, 135),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # F. WRITE CSV
        # ----------------------------------------------------

        writer.writerow([
            video_frame,
            round(timestamp, 3),
            required_atd_frame,
            raw_possession,
            final_possession,
            ball_x,
            ball_y,
            ball_speed,
            nearest_team,
            nearest_player_id,
            nearest_distance,
            len(visible_players),

            sum(
                team in ("Barcelona", "Athletic Club")
                for team in visible_players
            ),

            sum(
                role == "Goalkeeper?"
                for role in visible_roles
            ),

            sum(
                role == "Assistant Referee?"
                for role in visible_roles
            )
        ])

        # ----------------------------------------------------
        # G. DISPLAY
        # ----------------------------------------------------

        cv.imshow(
            "Barcelona vs Athletic - H/A/D/X Possession Analysis",
            frame
        )

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        if video_frame % 250 == 0:
            print(
                f"Processed {video_frame} frames | "
                f"Time: {timestamp:.2f}s | "
                f"Possession: {final_possession}"
            )

finally:

    capture.release()
    atd_file.close()
    csv_file.close()
    cv.destroyAllWindows()

# ============================================================
# 13. FINAL STATISTICS
# ============================================================

print("\n========================================")
print("       POSSESSION ANALYSIS COMPLETE")
print("========================================")

total_frames = sum(counts.values())

for label, description in [
    ("H", "Barcelona"),
    ("A", "Athletic Club"),
    ("D", "Dead ball"),
    ("X", "Unknown")
]:

    percentage = (
        counts[label] / total_frames * 100
        if total_frames > 0
        else 0
    )

    print(
        f"{description:15s}: "
        f"{counts[label]:8d} frames "
        f"({percentage:.2f}%)"
    )

valid_frames = counts["H"] + counts["A"]

print("\nValid possession frames:", valid_frames)

if valid_frames > 0:

    barcelona_percentage = (
        counts["H"] / valid_frames * 100
    )

    athletic_percentage = (
        counts["A"] / valid_frames * 100
    )

    print(
        f"Barcelona possession: "
        f"{barcelona_percentage:.2f}%"
    )

    print(
        f"Athletic Club possession: "
        f"{athletic_percentage:.2f}%"
    )

else:
    print("No valid possession observations available.")

print("\nCSV saved as:", OUTPUT_CSV)
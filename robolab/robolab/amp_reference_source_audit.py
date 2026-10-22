"""Explicit source-file keypoint meanings, not an automatic best-fit reorder.

The four start/stop clips store ankles, elbows, shoulders. The other four
store ankles, knees, elbows. Joint-angle order is the same. A full FK audit
against these declared link origins is mandatory before any R2R solve.
All TARGET datasets always export the policy's canonical six keypoints.
"""
CANONICAL = ("left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link",
             "right_knee_link", "left_elbow_yaw_link", "right_elbow_yaw_link")
START_STOP_STORED = ("left_ankle_roll_link", "right_ankle_roll_link",
                     "left_elbow_yaw_link", "right_elbow_yaw_link",
                     "left_arm_roll_link", "right_arm_roll_link")
START_STOP_CLIPS = frozenset(("run_start_180_R_001__A345_M", "run_start_180_R_001__A345",
                              "run_stop_180_R_001__A345_M", "run_stop_180_R_001__A345"))


def stored_source_keypoints(clip):
    return START_STOP_STORED if clip in START_STOP_CLIPS else CANONICAL

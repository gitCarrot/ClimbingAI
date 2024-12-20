import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture('climbing_video1.mp4')

BUFFER_SIZE = 20
MOVEMENT_THRESHOLD = 20

COG_BUFFER_SIZE = 5  
SUDDEN_JUMP_THRESHOLD = 100.0
BODY_CENTER_DISTANCE_THRESHOLD = 150.0

hand_left_buffer = deque(maxlen=BUFFER_SIZE)
hand_right_buffer = deque(maxlen=BUFFER_SIZE)
foot_left_buffer = deque(maxlen=BUFFER_SIZE)
foot_right_buffer = deque(maxlen=BUFFER_SIZE)

holds_positions = {
    'left_hand': [],
    'right_hand': [],
    'left_foot': [],
    'right_foot': []
}

hold_sequence_number = {
    'left_hand': 1,
    'right_hand': 1,
    'left_foot': 1,
    'right_foot': 1
}

cog_buffer = deque(maxlen=COG_BUFFER_SIZE)
prev_cog = None
pose_overlay_color = None

# Display toggles for each limb hold and pose lines
show_left_hand = True
show_right_hand = True
show_left_foot = True
show_right_foot = True
show_pose_lines = False  # If False, do not draw pose connections

# User-defined board angle (in degrees)
board_angle_deg = 50

# Pause/play state
paused = False
last_frame = None

def draw_pose(image, pose_landmarks, connections, color=(0,255,0), scale_factor=1.0, show_lines=True):
    if pose_landmarks is None or not show_lines:
        return
    # Draw pose connections
    for conn in connections:
        start = pose_landmarks.landmark[conn[0]]
        end = pose_landmarks.landmark[conn[1]]
        start_pt = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
        end_pt = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
        cv2.line(image, start_pt, end_pt, color, int(2*scale_factor))
    # Draw individual landmarks
    for lm in pose_landmarks.landmark:
        cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
        cv2.circle(image, (cx, cy), int(5*scale_factor), color, -1)

def is_stationary(buffer):
    # Checks if the limb position variation in the buffer is below threshold
    if len(buffer) < BUFFER_SIZE:
        return False
    xs, ys = zip(*buffer)
    if (max(xs)-min(xs)) < MOVEMENT_THRESHOLD and (max(ys)-min(ys)) < MOVEMENT_THRESHOLD:
        return True
    return False

def already_detected(holds_list, position):
    # Avoid recognizing the same hold position multiple times
    for hold in holds_list:
        dist = np.linalg.norm(np.array(hold['position']) - np.array(position))
        if dist < MOVEMENT_THRESHOLD * 2:
            return True
    return False

def draw_hold(image, pos, seq, label, color, scale_factor=1.0):
    # Draw a hold circle and sequence label
    radius = int(12*scale_factor)
    thickness = int(3*scale_factor)
    inner_color = tuple(int(c*0.7) for c in color)
    cv2.circle(image, pos, radius, inner_color, -1)
    cv2.circle(image, pos, radius, color, thickness)

    text = label + str(seq)
    font_scale = 0.7*scale_factor
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, int(2*scale_factor))
    bx, by = pos[0] + radius + 5, pos[1] - text_size[1]//2
    cv2.rectangle(image, (bx, by - 5), (bx + text_size[0] + 10, by + text_size[1] + 5), (60,60,60), -1)
    cv2.putText(image, text, (bx+5, by+text_size[1]), font, font_scale, (255,255,255), int(2*scale_factor))

def draw_cog(image, cog, scale_factor=1.0):
    # Draw circles indicating the CoG location
    for r in [30,20,10]:
        cv2.circle(image, (int(cog[0]), int(cog[1])), int(r*scale_factor), (200, 100, 200), int(2*scale_factor))
    cv2.circle(image, (int(cog[0]), int(cog[1])), int(10*scale_factor), (255, 0, 255), -1)
    cv2.circle(image, (int(cog[0]), int(cog[1])), int(10*scale_factor), (100, 0, 100), int(2*scale_factor))
    cv2.putText(image, 'CoG', (int(cog[0])+int(15*scale_factor), int(cog[1])+int(5*scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9*scale_factor, (255,255,255), int(2*scale_factor), cv2.LINE_AA)

def draw_arrow(image, start, end, color, scale_factor=1.0):
    # Draw tension direction arrow
    thickness = int(5*scale_factor)
    cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
    arrow_len = np.linalg.norm(np.array(end) - np.array(start))
    if arrow_len < 1e-6:
        return
    direction = (np.array(end) - np.array(start))/arrow_len
    perp = np.array([-direction[1], direction[0]])
    arrow_size = int(20*scale_factor)
    p1 = end
    p2 = end - direction*arrow_size + perp*(arrow_size/2)
    p3 = end - direction*arrow_size - perp*(arrow_size/2)
    pts = np.array([p1, p2, p3], dtype=np.int32)
    cv2.fillPoly(image, [pts], color)
    cv2.putText(image, 'Dir', (int(p2[0]), int(p2[1])-int(10*scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7*scale_factor, (255,255,255), int(2*scale_factor))

LEFT_INDEX_TIP = mp_holistic.HandLandmark.INDEX_FINGER_TIP.value
RIGHT_INDEX_TIP = mp_holistic.HandLandmark.INDEX_FINGER_TIP.value

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as holistic:
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        board_angle = math.radians(board_angle_deg)
        h, w, _ = frame.shape
        scale_factor = (w/640 + h/480) / 3.5

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Extract fingertip positions if available
        if results.left_hand_landmarks:
            lhm = results.left_hand_landmarks.landmark[LEFT_INDEX_TIP]
            left_hand_pos = (int(lhm.x * w), int(lhm.y * h))
        else:
            left_hand_pos = None

        if results.right_hand_landmarks:
            rhm = results.right_hand_landmarks.landmark[RIGHT_INDEX_TIP]
            right_hand_pos = (int(rhm.x * w), int(rhm.y * h))
        else:
            right_hand_pos = None

        # Extract toe positions if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_foot_index = mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value
            right_foot_index = mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value

            left_foot = (int(landmarks[left_foot_index].x * w),
                         int(landmarks[left_foot_index].y * h))
            right_foot = (int(landmarks[right_foot_index].x * w),
                          int(landmarks[right_foot_index].y * h))
        else:
            left_foot = None
            right_foot = None

        # Update buffers if not paused
        if not paused:
            if left_hand_pos is not None:
                hand_left_buffer.append(left_hand_pos)
            if right_hand_pos is not None:
                hand_right_buffer.append(right_hand_pos)
            if left_foot is not None:
                foot_left_buffer.append(left_foot)
            if right_foot is not None:
                foot_right_buffer.append(right_foot)

            # Check stationary limbs and record holds
            if is_stationary(hand_left_buffer):
                position = np.mean(hand_left_buffer, axis=0).astype(int)
                if not already_detected(holds_positions['left_hand'], position):
                    seq_num = hold_sequence_number['left_hand']
                    holds_positions['left_hand'].append({'position': tuple(position), 'sequence': seq_num})
                    hold_sequence_number['left_hand'] += 1

            if is_stationary(hand_right_buffer):
                position = np.mean(hand_right_buffer, axis=0).astype(int)
                if not already_detected(holds_positions['right_hand'], position):
                    seq_num = hold_sequence_number['right_hand']
                    holds_positions['right_hand'].append({'position': tuple(position), 'sequence': seq_num})
                    hold_sequence_number['right_hand'] += 1

            if is_stationary(foot_left_buffer):
                position = np.mean(foot_left_buffer, axis=0).astype(int)
                if not already_detected(holds_positions['left_foot'], position):
                    seq_num = hold_sequence_number['left_foot']
                    holds_positions['left_foot'].append({'position': tuple(position), 'sequence': seq_num})
                    hold_sequence_number['left_foot'] += 1

            if is_stationary(foot_right_buffer):
                position = np.mean(foot_right_buffer, axis=0).astype(int)
                if not already_detected(holds_positions['right_foot'], position):
                    seq_num = hold_sequence_number['right_foot']
                    holds_positions['right_foot'].append({'position': tuple(position), 'sequence': seq_num})
                    hold_sequence_number['right_foot'] += 1

        # Draw holds if toggled on
        if show_left_hand:
            for hold in holds_positions['left_hand']:
                draw_hold(image, hold['position'], hold['sequence'], "LH", (0,0,255), scale_factor)
        if show_right_hand:
            for hold in holds_positions['right_hand']:
                draw_hold(image, hold['position'], hold['sequence'], "RH", (0,255,0), scale_factor)
        if show_left_foot:
            for hold in holds_positions['left_foot']:
                draw_hold(image, hold['position'], hold['sequence'], "LF", (255,0,0), scale_factor)
        if show_right_foot:
            for hold in holds_positions['right_foot']:
                draw_hold(image, hold['position'], hold['sequence'], "RF", (0,255,255), scale_factor)

        if results.pose_landmarks:
            pose_overlay_color = (180,230,180)

            # Apply weighting based on board angle
            landmark_indices = {
                'left_shoulder': (mp_holistic.PoseLandmark.LEFT_SHOULDER.value, 0.08),
                'right_shoulder': (mp_holistic.PoseLandmark.RIGHT_SHOULDER.value, 0.08),
                'left_elbow': (mp_holistic.PoseLandmark.LEFT_ELBOW.value, 0.05),
                'right_elbow': (mp_holistic.PoseLandmark.RIGHT_ELBOW.value, 0.05),
                'left_wrist': (mp_holistic.PoseLandmark.LEFT_WRIST.value, 0.02),
                'right_wrist': (mp_holistic.PoseLandmark.RIGHT_WRIST.value, 0.02),
                'left_hip': (mp_holistic.PoseLandmark.LEFT_HIP.value, 0.15),
                'right_hip': (mp_holistic.PoseLandmark.RIGHT_HIP.value, 0.15),
                'left_knee': (mp_holistic.PoseLandmark.LEFT_KNEE.value, 0.18),
                'right_knee': (mp_holistic.PoseLandmark.RIGHT_KNEE.value, 0.18),
                'left_ankle': (mp_holistic.PoseLandmark.LEFT_ANKLE.value, 0.17),
                'right_ankle': (mp_holistic.PoseLandmark.RIGHT_ANKLE.value, 0.17),
            }

            board_angle = math.radians(board_angle_deg)
            hip_factor = 1.0 + abs(board_angle)
            ankle_factor = 1.0 + abs(board_angle)*0.5

            modified_landmark_indices = {}
            for name, (idx, weight) in landmark_indices.items():
                new_weight = weight
                if 'hip' in name:
                    new_weight = weight * hip_factor
                elif 'ankle' in name:
                    new_weight = weight * ankle_factor
                modified_landmark_indices[name] = (idx, new_weight)

            landmarks = results.pose_landmarks.landmark
            keypoints = []
            weight_factors = []
            all_points = []
            for name, (idx, weight) in modified_landmark_indices.items():
                lm = landmarks[idx]
                x = lm.x * w
                y = lm.y * h
                all_points.append((x, y))
                if lm.visibility > 0.5:
                    keypoints.append(np.array([x, y]))
                    weight_factors.append(weight)

            all_points = np.array(all_points)
            body_center = np.mean(all_points, axis=0)

            # Compute CoG and handle sudden jumps
            if len(keypoints) > 0:
                keypoints = np.array(keypoints)
                weight_factors = np.array(weight_factors)
                new_cog = np.average(keypoints, axis=0, weights=weight_factors)

                if not paused:
                    cog_buffer.append(new_cog)
                cog_smoothed = np.mean(cog_buffer, axis=0) if len(cog_buffer) > 0 else new_cog

                sudden_jump = False
                dist_shift = 0
                if prev_cog is not None:
                    dist_shift = np.linalg.norm(cog_smoothed - prev_cog)
                    dist_from_body_center = np.linalg.norm(cog_smoothed - body_center)
                    if dist_shift > SUDDEN_JUMP_THRESHOLD and dist_from_body_center > BODY_CENTER_DISTANCE_THRESHOLD:
                        sudden_jump = True
                prev_cog = cog_smoothed.copy()

                if len(cog_buffer) > 1:
                    prev_pos = np.mean(list(cog_buffer)[:-1], axis=0)
                    cv2.line(image,
                             (int(prev_pos[0]), int(prev_pos[1])),
                             (int(cog_smoothed[0]), int(cog_smoothed[1])),
                             (200, 100, 200), int(2*scale_factor))

                draw_cog(image, cog_smoothed, scale_factor)

                if sudden_jump:
                    pose_overlay_color = (255,180,180)

                # Compute limb-based tension direction
                limb_positions = []
                if left_hand_pos is not None:
                    limb_positions.append(np.array(left_hand_pos))
                if right_hand_pos is not None:
                    limb_positions.append(np.array(right_hand_pos))
                if left_foot is not None:
                    limb_positions.append(np.array(left_foot))
                if right_foot is not None:
                    limb_positions.append(np.array(right_foot))

                if len(limb_positions) > 0:
                    limb_vectors = [pos - body_center for pos in limb_positions]
                    limb_sum_vec = np.sum(limb_vectors, axis=0)
                else:
                    limb_sum_vec = np.array([0.0,0.0])

                direction_vec_from_limbs = -limb_sum_vec
                direction_len = np.linalg.norm(direction_vec_from_limbs)
                if direction_len > 1e-6:
                    direction_vec_from_limbs = direction_vec_from_limbs / direction_len
                else:
                    direction_vec_from_limbs = np.array([0,0])

                gravity_vec = np.array([0,1.0])
                threshold = 100
                if direction_len < threshold:
                    # If limbs are close, use gravity-only direction
                    combined_vec = gravity_vec
                else:
                    combined_vec = gravity_vec + direction_vec_from_limbs
                    c_len = np.linalg.norm(combined_vec)
                    if c_len > 1e-6:
                        combined_vec = combined_vec / c_len
                    else:
                        combined_vec = gravity_vec

                base_arrow_length = 80*scale_factor
                speed_factor = 5.0*scale_factor
                arrow_length = base_arrow_length + speed_factor * dist_shift

                start_pt = (int(cog_smoothed[0]), int(cog_smoothed[1]))
                end_pt = (int(cog_smoothed[0] + combined_vec[0]*arrow_length),
                          int(cog_smoothed[1] + combined_vec[1]*arrow_length))

                draw_arrow(image, start_pt, end_pt, (255,128,0), scale_factor)

            draw_pose(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                      color=pose_overlay_color, scale_factor=scale_factor, show_lines=show_pose_lines)

            cv2.putText(image, f'Board Angle: {board_angle_deg:.1f} deg', (30,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0*scale_factor, (255,255,255), int(2*scale_factor))

        cv2.imshow('Climber Holds and Center of Gravity', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            show_left_hand = not show_left_hand
        elif key == ord('2'):
            show_right_hand = not show_right_hand
        elif key == ord('3'):
            show_left_foot = not show_left_foot
        elif key == ord('4'):
            show_right_foot = not show_right_foot
        elif key == ord('p'):
            show_pose_lines = not show_pose_lines
        elif key == ord('['):
            board_angle_deg -= 5
        elif key == ord(']'):
            board_angle_deg += 5
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
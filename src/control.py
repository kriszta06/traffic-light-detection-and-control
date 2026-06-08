"""
Description: Implements simulated vehicle reaction logic based on detected traffic light states.
"""


class VehicleControlUnit:

    def __init__(self):
        self.current_state        = "GO"
        self.min_confirm_frames   = 4
        self.consecutive_count    = 0   # consecutive frames with a valid light in corridor
        self.max_distance         = 60.0
        self.area_ref             = 50000.0
        self.area_max             = 1000000.0
        self.stop_area_pixels     = 20000.0
        self.max_stop_distance    = 5.0

    def select_relevant_traffic_light(self, tracked_objects, lane_info=None):
        if not tracked_objects:
            self.consecutive_count = 0
            return None

        if lane_info and 'image_shape' in lane_info:
            height, width = lane_info['image_shape'][:2]
        else:
            width, height = 1280, 720

        lane_center_x  = width / 2.0
        corridor_half  = width * 0.18
        corridor_left  = lane_center_x - corridor_half
        corridor_right = lane_center_x + corridor_half

        valid_candidates = []
        for obj in tracked_objects:
            box = obj.get('box', [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = box
            center_x  = (xmin + xmax) / 2.0
            box_width  = max(0.0, xmax - xmin)
            box_height = max(0.0, ymax - ymin)
            area = box_width * box_height

            if center_x < corridor_left or center_x > corridor_right:
                continue
            cls = obj.get('class', 'Traffic_Light')
            min_area = 10.0 if cls in ('Red_Light', 'Yellow_Light') else 200.0
            if area < min_area:
                continue
            priority = 0 if cls in ('Red_Light', 'Yellow_Light') else 1
            valid_candidates.append((priority, -area, ymin, abs(center_x - lane_center_x), obj))

        chosen = None

        if not valid_candidates:
            # Wider fallback: only Red/Yellow within +-35% of center.
            wider_half  = width * 0.35
            wider_left  = lane_center_x - wider_half
            wider_right = lane_center_x + wider_half
            alt_candidates = []
            for obj in tracked_objects:
                box = obj.get('box', [0, 0, 0, 0])
                xmin, ymin, xmax, ymax = box
                center_x = (xmin + xmax) / 2.0
                area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                cls  = obj.get('class', 'Traffic_Light')
                if (cls in ('Red_Light', 'Yellow_Light')
                        and area >= 30.0
                        and wider_left <= center_x <= wider_right):
                    alt_candidates.append((-area, ymin, obj))
            if alt_candidates:
                alt_candidates.sort(key=lambda item: (item[0], item[1]))
                chosen = alt_candidates[0][2]
        else:
            valid_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            chosen = valid_candidates[0][4]

        if chosen is None:
            self.consecutive_count = 0
            return None

        # Consecutive-frame confirmation: counts how many frames in a row a valid
        # light has been present in the corridor. Robust to position shifts (approaching
        # traffic light moves in frame) unlike the old 8px spatial-bucket approach.
        self.consecutive_count += 1
        chosen['confirmed'] = bool(self.consecutive_count >= self.min_confirm_frames)
        return chosen

    def generate_command(self, target_light):
        control = {
            'action': 'GO', 'throttle': 0.6, 'brake': 0.0,
            'handbrake': False, 'distance_to_light': float('inf')
        }

        if not target_light:
            return control

        state = target_light.get('class', 'Unknown')
        xmin, ymin, xmax, ymax = target_light.get('box', [0.0, 0.0, 0.0, 0.0])
        area = float((xmax - xmin) * (ymax - ymin))

        if area < 15.0:
            return control

        size_factor = min(1.0, area / self.area_ref)

        # Inverse-sqrt distance formula (perspective law): C/sqrt(area)
        # Calibrated so area=10000px -> ~10m, area=40000px -> ~5m (stop threshold).
        C = 1000.0
        distance_to_light = min(self.max_distance, max(1.0, C / max(area ** 0.5, 1.0)))

        control['distance_to_light'] = distance_to_light

        confirmed_flag = bool(target_light.get('confirmed', False))

        print(f"VCU DEBUG: state={state}, area={area:.1f}, dist={distance_to_light:.2f}m, "
              f"confirmed={confirmed_flag} (streak={self.consecutive_count})")

        # Safety fallback: STOP only when confirmed AND at close range.
        if confirmed_flag and distance_to_light <= self.max_stop_distance and state in ('Red_Light', 'Yellow_Light'):
            control['action']    = 'STOP'
            control['throttle']  = 0.0
            control['brake']     = 1.0
            control['handbrake'] = True
            control['distance_to_light'] = min(distance_to_light, 5.0)
            return control

        if state == 'Red_Light':
            if confirmed_flag and distance_to_light <= self.max_stop_distance:
                control['action']   = 'STOP'
                control['throttle'] = 0.0
                control['brake']    = 1.0
            else:
                control['action']   = 'CAUTION'
                control['throttle'] = max(0.0, 0.35 * (1.0 - size_factor))
                control['brake']    = min(1.0, 0.25 * size_factor)

        elif state == 'Yellow_Light':
            if confirmed_flag and distance_to_light <= self.max_stop_distance:
                control['action']   = 'STOP'
                control['throttle'] = 0.0
                control['brake']    = 1.0
            else:
                control['action']   = 'CAUTION'
                control['throttle'] = max(0.0, 0.4 * (1.0 - size_factor))
                control['brake']    = min(1.0, 0.15 + 0.65 * size_factor)

        elif state == 'Green_Light':
            control['action']   = 'GO'
            control['throttle'] = 0.4 if area > 400.0 else min(1.0, 0.5 + 0.35 * size_factor)
            control['brake']    = 0.0

        else:
            # Generic Traffic_Light (unknown color) — caution but never STOP
            if area > 200.0:
                control['action']   = 'CAUTION'
                control['throttle'] = 0.2
                control['brake']    = 0.2
            else:
                control['action']   = 'GO'
                control['throttle'] = max(0.0, 0.55 - 0.25 * size_factor)
                control['brake']    = min(1.0, 0.1 + 0.25 * size_factor)

        control['handbrake'] = control['brake'] > 0.9
        return control

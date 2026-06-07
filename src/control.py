"""
Description: Implements simulated vehicle reaction logic based on detected traffic light states.
"""

class VehicleControlUnit:
    """
    A core decision class that acts as the interface to the simulator's vehicle ego-control API.
    """
    def __init__(self):
        """
        Initializes the vehicle's internal state machine.
        """
        self.current_state = "GO"
        # temporal confirmation counts for detected objects (bucketed center keys)
        self.confirm_counts = {}
        self.confirm_tolerance = 8.0
        self.min_confirm_frames = 4
        # mapping parameters: estimated max distance (meters) for smallest detections
        self.max_distance = 60.0
        # reference area used to compute size_factor; smaller value -> distance decreases earlier
        # reference area used to compute size_factor; larger value -> distance decreases later
        self.area_ref = 50000.0
        self.area_max = 1000000.0
        self.stop_area_pixels = 20000.0
        self.max_stop_distance = 5.0

    def select_relevant_traffic_light(self, tracked_objects, lane_info=None):
        """
        Filters out irrelevant traffic lights to isolate the single light governing the vehicle's active driving path.

        Parameters:
            tracked_objects (list): Active tracked traffic lights in the current scene.
            lane_info (dict, optional): Lane keeping or waypoint data.

        Returns:
            dict: The target traffic light object that the vehicle must obey, or None if no light affects the current lane.
        """
        if not tracked_objects:
            # prune counters
            self.confirm_counts.clear()
            return None

        if lane_info and 'image_shape' in lane_info:
            height, width = lane_info['image_shape'][:2]
        else:
            width, height = 1280, 720

        lane_center_x = width / 2.0
        corridor_half = width * 0.18
        corridor_left = lane_center_x - corridor_half
        corridor_right = lane_center_x + corridor_half

        valid_candidates = []
        seen_keys = set()
        for obj in tracked_objects:
            box = obj.get('box', [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = box
            center_x = (xmin + xmax) / 2.0
            box_width = max(0.0, xmax - xmin)
            box_height = max(0.0, ymax - ymin)
            area = box_width * box_height

            # update confirmation bucket
            key = (int(center_x // self.confirm_tolerance), int(((ymin + ymax) / 2.0) // self.confirm_tolerance))
            seen_keys.add(key)
            self.confirm_counts[key] = self.confirm_counts.get(key, 0) + 1

            if center_x < corridor_left or center_x > corridor_right:
                continue
            cls = obj.get('class', 'Traffic_Light')
            min_area = 10.0 if cls in ('Red_Light', 'Yellow_Light') else 200.0
            if area < min_area:
                continue
            priority = 0 if cls in ('Red_Light', 'Yellow_Light') else 1
            valid_candidates.append((priority, -area, ymin, abs(center_x - lane_center_x), obj))

        # prune unseen keys
        for k in list(self.confirm_counts.keys()):
            if k not in seen_keys:
                del self.confirm_counts[k]

        if not valid_candidates:
            # If nothing in the forward corridor was found, prefer any red/yellow light in the frame
            alt_candidates = []
            for obj in tracked_objects:
                box = obj.get('box', [0, 0, 0, 0])
                xmin, ymin, xmax, ymax = box
                area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                cls = obj.get('class', 'Traffic_Light')
                if cls in ('Red_Light', 'Yellow_Light') and area >= 8.0:
                    # prefer larger area and closer to bottom
                    alt_candidates.append((-area, ymin, obj))
            if alt_candidates:
                alt_candidates.sort(key=lambda item: (item[0], item[1]))
                chosen = alt_candidates[0][2]
                xmin, ymin, xmax, ymax = chosen.get('box', [0, 0, 0, 0])
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                key = (int(center_x // self.confirm_tolerance), 
                       int(center_y // self.confirm_tolerance))
                confirmed = self.confirm_counts.get(key, 0) >= self.min_confirm_frames
                chosen['confirmed'] = bool(confirmed)
                return chosen
            return None

        valid_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        chosen = valid_candidates[0][4]
        xmin, ymin, xmax, ymax = chosen.get('box', [0, 0, 0, 0])
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        key = (int(center_x // self.confirm_tolerance), 
               int(center_y // self.confirm_tolerance))
        confirmed = self.confirm_counts.get(key, 0) >= self.min_confirm_frames
        chosen['confirmed'] = self.confirm_counts.get(key, 0) >= self.min_confirm_frames
        return chosen

    def generate_command(self, target_light):
            """
            Translates the classified color state of the relevant traffic light into discrete or continuous control signals.

            Parameters:
                target_light (dict): The filtered target traffic light containing its current state.

            Returns:
                dict: A control payload containing actuation values (e.g., action: str, throttle: float, brake: float).
            """
            # Comanda implicită în caz că nu există semafor în zonă
            control = {'action': 'GO', 'throttle': 0.6, 'brake': 0.0, 'handbrake': False, 'distance_to_light': float('inf')}

            if not target_light:
                return control
            
            state = target_light.get('class', 'Unknown')
            xmin, ymin, xmax, ymax = target_light.get('box', [0.0, 0.0, 0.0, 0.0])
            area = float((xmax - xmin) * (ymax - ymin))

            # FILTRU CRITIC DE ZGOMOT: Ignorăm artefactele minuscule (sub 15px) ca să nu frânăm aiurea
            if area < 15.0:
                return control

            size_factor = min(1.0, area / self.area_ref)
            distance_to_light = max(2.0, 60.0 * (1.0 - (area / 2000.0)**0.3))
            
            # Ajustare distanță empirică dacă obiectul ocupă mult spațiu pe ecran
            if area > 800.0:
                distance_to_light = min(distance_to_light, 4.5)

            control['distance_to_light'] = distance_to_light

            try:
                confirmed_flag = bool(target_light.get('confirmed', False))
            except Exception:
                confirmed_flag = False

            # Debug corectat (am eliminat 'area_clamped' care genera NameError)
            print(f"VCU DEBUG: state={state}, area={area:.1f}, size_factor={size_factor:.3f}, distance={distance_to_light:.2f}, confirmed={confirmed_flag}")

            # =========================================================================
            # LOGICA DE DECIZIE UNIFICATĂ (Fără riscuri de suprascriere)
            # =========================================================================
            
            # 1. COMANDA DE URGENȚĂ (Safety Fallback): Dacă suntem extrem de aproape sau semaforul e imens
            if area > 600.0 or distance_to_light <= self.max_stop_distance:
                # Dacă e roșu/galben SAU dacă e atât de aproape încât clasificatorul de culoare dă rateuri (ex: Green_Light din greșeală)
                if state in ['Red_Light', 'Yellow_Light', 'Traffic_Light'] or area > 600.0:
                    control['action'] = 'STOP'
                    control['throttle'] = 0.0
                    control['brake'] = 1.0
                    control['handbrake'] = True
                    control['distance_to_light'] = min(distance_to_light, 5.0) # Forțăm afișarea distanței de oprire
                    return control # Oprim execuția aici ca să nu poată fi suprascrisă!

            # 2. LOGICA DE DEPLASARE NORMALĂ (Când semaforul e încă la distanță sigură)
            if state == 'Red_Light':
                if confirmed_flag and distance_to_light <= self.max_stop_distance:
                    control['action'] = 'STOP'
                    control['throttle'] = 0.0
                    control['brake'] = 1.0
                else:
                    control['action'] = 'CAUTION'
                    control['throttle'] = max(0.0, 0.35 * (1.0 - size_factor))
                    control['brake'] = min(1.0, 0.25 * size_factor)

            elif state == 'Yellow_Light':
                if confirmed_flag and distance_to_light <= self.max_stop_distance:
                    control['action'] = 'STOP'
                    control['throttle'] = 0.0
                    control['brake'] = 1.0
                else:
                    control['action'] = 'CAUTION'
                    control['throttle'] = max(0.0, 0.4 * (1.0 - size_factor))
                    control['brake'] = min(1.0, 0.15 + 0.65 * size_factor)

            elif state == 'Green_Light':
                control['action'] = 'GO'
                # Dacă semaforul e măricel, reducem puțin viteza preventiv
                control['throttle'] = 0.4 if area > 400.0 else min(1.0, 0.5 + 0.35 * size_factor)
                control['brake'] = 0.0

            else: # Pentru clasa generică 'Traffic_Light' fără culoare detectată clar de masca HSV
                if area > 200.0:
                    control['action'] = 'CAUTION'
                    control['throttle'] = 0.2
                    control['brake'] = 0.2
                else:
                    control['action'] = 'GO'
                    control['throttle'] = max(0.0, 0.55 - 0.25 * size_factor)
                    control['brake'] = min(1.0, 0.1 + 0.25 * size_factor)

            # Activăm frâna de mână automată la decelerări masive
            control['handbrake'] = control['brake'] > 0.9
            return control

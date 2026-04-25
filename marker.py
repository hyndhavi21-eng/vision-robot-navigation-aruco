import cv2
import numpy as np
import heapq

# 🔴 IP Webcam
ip_url = "http://192.168.1.16:8080/video"

ALPHA = 0.2
prev_points = None

BOUNDARY_HOLD_FRAMES = 10
boundary_lost = 0

# 🧠 ArUco
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

prev_bot = None
prev_goal = None

# 🔥 Obstacle params
MIN_OBSTACLE_AREA = 800
MAX_OBSTACLE_AREA = 15000

# 🔥 Grid size
GRID_SIZE = 20


# ================= A* =================
def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    def h(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            n = (current[0]+dx, current[1]+dy)

            if 0 <= n[0] < rows and 0 <= n[1] < cols:
                if grid[n] == 1:
                    continue

                new_g = g_score[current] + 1

                if n not in g_score or new_g < g_score[n]:
                    g_score[n] = new_g
                    f = new_g + h(n, goal)
                    heapq.heappush(open_set, (f, n))
                    came_from[n] = current

    return []


# ================= FUNCTIONS =================

def detect_blue_corners(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array([90,120,60]),
                       np.array([130,255,255]))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    cv2.imshow("Blue Mask", mask)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    pts=[]
    for c in contours:
        if cv2.contourArea(c)>100:
            (x,y),_ = cv2.minEnclosingCircle(c)
            pts.append((int(x),int(y)))
    return pts


def get_stable_corners(points):
    pts = np.array(points)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ])


def smooth_point(new, prev):
    if prev is None:
        return new
    return (int(ALPHA*new[0] + (1-ALPHA)*prev[0]),
            int(ALPHA*new[1] + (1-ALPHA)*prev[1]))


def draw_dotted_line(img, pt1, pt2, color, gap=10):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((1 - r) * pt1[0] + r * pt2[0])
        y = int((1 - r) * pt1[1] + r * pt2[1])
        cv2.circle(img, (x, y), 2, color, -1)


# ================= MAIN =================

cap = cv2.VideoCapture(ip_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bot_front = None
    goal_point = None

    # ================= BOUNDARY =================
    points = detect_blue_corners(frame)

    if len(points) >= 4:
        ordered = get_stable_corners(points)

        if prev_points is None:
            prev_points = ordered
        else:
            ordered = np.array([
                smooth_point(tuple(ordered[i]), tuple(prev_points[i]))
                for i in range(4)
            ])
            prev_points = ordered

        cv2.polylines(frame, [ordered], True, (0,255,0), 3)

    # ================= ARUCO =================
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            c = corners[i][0]

            if marker_id == 1:
                cx, cy = int(np.mean(c[:,0])), int(np.mean(c[:,1]))
                prev_bot = smooth_point((cx,cy), prev_bot)

                fx = int((c[2][0]+c[3][0])/2)
                fy = int((c[2][1]+c[3][1])/2)
                bot_front = (fx, fy)

                cv2.circle(frame, prev_bot, 3, (0,0,255), -1)
                cv2.circle(frame, bot_front, 4, (255,0,0), -1)

                cv2.putText(frame, "BOT",
                    (max(prev_bot[0]-30,0), max(prev_bot[1]-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 2)

            elif marker_id == 0:
                cx, cy = int(np.mean(c[:,0])), int(np.mean(c[:,1]))
                prev_goal = smooth_point((cx,cy), prev_goal)
                goal_point = prev_goal

                cv2.circle(frame, goal_point, 4, (0,255,0), -1)

                cv2.putText(frame, "GOAL",
                (max(goal_point[0]-30,0), max(goal_point[1]-10,0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2)

    # ================= OBSTACLE DETECTION =================
    edges = cv2.Canny(gray,50,150)
    edges = cv2.dilate(edges,np.ones((3,3),np.uint8),1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))

    if ids is not None:
        for i in range(len(ids)):
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask,[corners[i][0].astype(int)],255)
            mask = cv2.dilate(mask,np.ones((25,25),np.uint8))
            edges[mask==255] = 0

    if prev_points is not None:
        arena = np.zeros_like(edges)
        cv2.fillPoly(arena,[prev_points.astype(int)],255)
        edges = cv2.bitwise_and(edges, arena)

    cv2.imshow("Edges Clean", edges)

    # ================= GRID =================
    h,w = frame.shape[:2]
    grid = np.zeros((h//GRID_SIZE, w//GRID_SIZE), dtype=np.uint8)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_OBSTACLE_AREA:
            continue

        x,y,wc,hc = cv2.boundingRect(cnt)
        cx,cy = x+wc//2, y+hc//2

        gx,gy = cx//GRID_SIZE, cy//GRID_SIZE

        # obstacle inflation
        for dx in range(-1,2):
            for dy in range(-1,2):
                nx,ny = gx+dx, gy+dy
                if 0<=ny<grid.shape[0] and 0<=nx<grid.shape[1]:
                    grid[ny,nx]=1

        cv2.rectangle(frame,(x,y),(x+wc,y+hc),(0,0,255),2)

    # ================= A* PATH =================
    if bot_front is not None and goal_point is not None:
        start = (bot_front[1]//GRID_SIZE, bot_front[0]//GRID_SIZE)
        goal = (goal_point[1]//GRID_SIZE, goal_point[0]//GRID_SIZE)

        path = astar(grid, start, goal)

        if path:
            for i in range(len(path)-1):
                y1,x1 = path[i]
                y2,x2 = path[i+1]

                p1 = (x1*GRID_SIZE, y1*GRID_SIZE)
                p2 = (x2*GRID_SIZE, y2*GRID_SIZE)

                draw_dotted_line(frame, p1, p2, (0,255,255))

                # ================= NEXT WAYPOINT =================
        if len(path) > 3:
            next_node = path[3]   # look-ahead

            ny, nx = next_node
            next_point = (nx * GRID_SIZE, ny * GRID_SIZE)

            # 🔥 draw next target
            cv2.circle(frame, next_point, 6, (255,0,255), -1)

            # ================= DIRECTION =================
            dir_vec = np.array(bot_front) - np.array(prev_bot)
            target_vec = np.array(next_point) - np.array(prev_bot)

            # ================= ANGLE =================
            angle = np.arctan2(dir_vec[1], dir_vec[0]) - np.arctan2(target_vec[1], target_vec[0])
            angle = np.degrees(angle)

            # normalize angle
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360

            # ================= DECISION =================
            if abs(angle) < 10:
                action = "FORWARD"
            elif angle > 0:
                action = "LEFT"
            else:
                action = "RIGHT"

            # ================= STOP CONDITION =================
            dist_to_goal = np.linalg.norm(np.array(prev_bot) - np.array(goal_point))
            if dist_to_goal < 20:
                action = "STOP"

            # ================= DISPLAY =================
            cv2.putText(frame, f"{action}", (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.putText(frame, f"Angle: {int(angle)}", (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.putText(frame, f"Dist: {int(dist_to_goal)}", (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Robot View", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
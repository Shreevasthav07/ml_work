from collections import deque, heapq

def valid(s):
    F, W, G, C = s
    if ((W == G != F ) or (G==C != F)):
        return False
    return True

def moves(s):
    F, W, G, C = s
    next_states =[]

    #farmer moves alone
    if F == 0:
        newF = 1
    else:
        newF = 0
    new_state = (newF, W, G, C)
    if valid(new_state):
        next_states.append(new_state)
    
    #farmer takes wolf
    if F == W:
        newF = 1 if F == 0 else 0
        newW = 1 if W ==0 else 0
        new_state = (newF, newW,G,C)
        if valid(new_state):
            next_states.append(new_state)
    
    # farmer takes goat
    if F == G:
        newF = 1 if F ==0 else 0
        newG = 1 if G ==0 else 0
        new_state =  (newF,W,newG,C)
        if valid(new_state):
            next_states.append(new_state)
    
    # farmer takes cabbage
    if F == C:
        newF = 1 if F == 0 else 0
        newC = 1 if C == 0 else 0
        new_state = (newF, W, G,newC)
        if valid(new_state):
            next_states.append(new_state)
    
    return next_states

def bfs(start,goal):
    queue = deque([(start,[start])])
    visited = {start}

    while queue:
        state, path = queue.popleft()
        if state ==goal:
            return path
        for next_state in moves(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state,path+[next_state]))
    return None

def dfs(start,goal):
    stack = [(start,[start])]
    visited = {start}

    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        for next_state in moves(state):
            if next_state not in visited:
                visited.add(next_state)
                stack.append((next_state,path+[next_state]))
    return None

def dls(start,goal,limit):
    stack = [(start,[start],0)]
    while stack:
        state,path,depth = stack.pop()
        if state == goal:
            return path
        if depth<limit:
            for next_state in moves(state):
                if next_state not in path:
                    stack.append((next_state,path+[next_state],depth+1))
    return None

def ids(start,goal,max_limit =20):
    for limit in range(max_limit+1):
        print(f"Trying Depth Limit = {limit}.....")
        result = dls(start,goal,limit)
        if result is not None:
            print(f"Goal found at depth : {limit}")
            return result
    return None

    
def uniform_cost_search(start, goal):
    # cost for each move is 1 (could be changed)
    pq = []  # elements are (cost, state)
    heapq.heappush(pq, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while pq:
        cost, state = heapq.heappop(pq)
        if state == goal:
            # reconstruct path
            path = []
            cur = state
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        for nxt in moves(state):
            new_cost = cost_so_far[state] + 1  # every action cost = 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = state
                heapq.heappush(pq, (new_cost, nxt))
    return None

# --------------------------
# Bidirectional Search
# --------------------------
def reconstruct_bidirectional(meet_node, parents_fwd, parents_bwd):
    # build forward path from start to meet_node
    path_fwd = []
    cur = meet_node
    while cur is not None:
        path_fwd.append(cur)
        cur = parents_fwd.get(cur, None)
    path_fwd.reverse()  # now start -> meet_node

    # build backward path from meet_node to goal (exclude meet_node to avoid duplicate)
    path_bwd = []
    cur = parents_bwd.get(meet_node, None)  # node after meet_node in backward tree
    while cur is not None:
        path_bwd.append(cur)
        cur = parents_bwd.get(cur, None)

    return path_fwd + path_bwd

def bidirectional_search(start, goal):
    if start == goal:
        return [start]

    # frontier queues
    q_fwd = deque([start])
    q_bwd = deque([goal])
    parents_fwd = {start: None}
    parents_bwd = {goal: None}
    visited_fwd = {start}
    visited_bwd = {goal}

    while q_fwd and q_bwd:
        # expand one level forward
        for _ in range(len(q_fwd)):
            s = q_fwd.popleft()
            for nxt in moves(s):
                if nxt not in visited_fwd:
                    parents_fwd[nxt] = s
                    visited_fwd.add(nxt)
                    q_fwd.append(nxt)
                    if nxt in visited_bwd:
                        # meet point
                        return reconstruct_bidirectional(nxt, parents_fwd, parents_bwd)

        # expand one level backward
        for _ in range(len(q_bwd)):
            s = q_bwd.popleft()
            # moves(s) gives legal moves from s (i.e., successors). For backward expansion we need predecessors.
            # Because the transition relation is symmetric (moving farmer +/- item toggles bits),
            # successors and predecessors sets are identical here — we can reuse moves(s).
            for nxt in moves(s):
                if nxt not in visited_bwd:
                    parents_bwd[nxt] = s
                    visited_bwd.add(nxt)
                    q_bwd.append(nxt)
                    if nxt in visited_fwd:
                        # meet point
                        return reconstruct_bidirectional(nxt, parents_fwd, parents_bwd)

    return None

# --------------------------
# A* Search
# --------------------------
def heuristic(state, goal):
    # simple admissible heuristic: number of items (wolf, goat, cabbage) not yet at goal side
    # indices 1,2,3 correspond to W,G,C; farmer at index 0
    # each move can move at most one item, so we need at least that many moves (admissible).
    # This heuristic ignores that farmer has to return sometimes, so it's admissible (often weak).
    return abs(state[1] - goal[1]) + abs(state[2] - goal[2]) + abs(state[3] - goal[3])

def a_star(start, goal):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))  # (f = g+h, g, state)
    came_from = {start: None}
    g_score = {start: 0}
    closed = set()

    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        if state == goal:
            # reconstruct
            path = []
            cur = state
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if state in closed:
            continue
        closed.add(state)

        for nxt in moves(state):
            tentative_g = g_score[state] + 1
            if nxt in closed and tentative_g >= g_score.get(nxt, float('inf')):
                continue
            if tentative_g < g_score.get(nxt, float('inf')):
                came_from[nxt] = state
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt))
    return None


start = (0,0,0,0)
goal = (1,1,1,1)
bfs_solution = bfs(start, goal)
dfs_solution = dfs(start,goal)
dls_solution = dls(start,goal,50)
ids_solution = ids(start,goal,20)

print("=====BFS Solution ========")
for step, state in enumerate(bfs_solution):
    print(f"Step: {step+1}, Farmer={state[0]}, Wolf={state[1]}, Goat={state[2]}, Cabbage={state[3]}")

print("\n=====DFS Solution=====")
for step, state in enumerate(dfs_solution):
    print(f"Step: {step+1},Farmer={state[0]}, Wolf={state[1]}, Goat={state[2]}, Cabbage={state[3]}")

if dls_solution is not None:
    print("\n=====DLS Solution=====")
    for step, state in enumerate(dls_solution):
        print(f"Step: {step+1},Farmer={state[0]}, Wolf={state[1]}, Goat={state[2]}, Cabbage={state[3]}")
else:
    print("\nNo solution found within depth limit.")

if ids_solution is not None:
    print("\n=====IDS Solution=====")
    for step, state in enumerate(ids_solution):
        print(f"Step: {step+1},Farmer={state[0]}, Wolf={state[1]}, Goat={state[2]}, Cabbage={state[3]}")
else:
    print("\nNo solution found within MAX depth limit.")


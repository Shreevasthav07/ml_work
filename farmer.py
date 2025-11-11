from collections import deque

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


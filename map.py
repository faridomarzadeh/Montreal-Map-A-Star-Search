import os

import matplotlib.pyplot as plt
import numpy as np
import shapefile
import networkx as nx
from heapq import heappush, heappop
from itertools import count
from matplotlib.patches import Rectangle

class GRID:
    x_axis=0
    y_axis=0
    total=0
    type=0
    def getKey(self):
        return str(self.x_axis)+str(self.y_axis)
class EDGE:
    def __init__(self,x1,y1,x2,y2):
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
    cost=0
    g1=None
    g2=None
    isEdge=False
def astar_path(G, source, target, heuristic=None, weight='weight'):

    push = heappush
    pop = heappop
    c = count()
    queue = [(0, next(c), source, 0, None)]

    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            if neighbor in explored:
                continue
            ncost = dist + w.get(weight, 1)
            if ncost>=1000000:
                continue
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

def add_edge(x1,x2,y1,y2,grid):
    keys=[]
    local_edges=[]
    key1=str(x1)+str(y1)+str(x2)+str(y2)
    edge1=EDGE(x1,y1,x2,y2) #diagnol
    key2=str(x1)+str(y1)+str(x2)+str(y1)
    edge2=EDGE(x1,y1,x2,y1)
    local_edges.append(edge2)
    keys.append(key2)
    key3=str(x1)+str(y1)+str(x1)+str(y2)
    edge3=EDGE(x1,y1,x1,y2)
    local_edges.append(edge3)
    keys.append(key3)
    key4=str(x1)+str(y2)+str(x2)+str(y1)
    edge4=EDGE(x1,y2,x2,y1) #diagnol
    key5=str(x1)+str(y2)+str(x2)+str(y2)
    edge5=EDGE(x1,y2,x2,y2)
    local_edges.append(edge5)
    keys.append(key5)
    key6=str(x2)+str(y1)+str(x2)+str(y2)
    edge6=EDGE(x2,y1,x2,y2)
    local_edges.append(edge6)
    keys.append(key6)

    edges[key1]=edge1
    edges[key4]=edge4
    edges[key1].g1=grid
    edges[key1].g2=grid
    edges[key4].g1=grid
    edges[key4].g2=grid
    k=0
    #print(keys)
    for key in keys:
        if key in edges.keys():
            edges[key].g2=grid
        else:
            if str(local_edges[k].x1) == str(local_edges[k].x2)==str(x_min) or str(local_edges[k].x1) == str(local_edges[k].x2)==str(x_max) or str(local_edges[k].y1)==str(local_edges[k].y2)==str(y_min) or str(local_edges[k].y1)==str(local_edges[k].y2)==str(y_max):
                edges[key] = local_edges[k]
                edges[key].g1 = grid
                edges[key].g2 = grid
                edges[key].isEdge = True
            else:
                edges[key] = local_edges[k]
                edges[key].g1 = grid
        k+=1

x_min=-73.59
x_max=-73.55
y_min=45.49
y_max=45.53
num=float(input('enter size\n'))
threshold=float(input("threshold:\n"))
threshold*=100;
print((x_max-x_min)/num)
print((y_max-y_min)/num)
xi=int(round((x_max-x_min)/num))
yi=int(round((y_max-y_min)/num))
num=round((x_max-x_min)/xi,5)
grid=[[GRID() for j in range(yi)] for i in range(xi)]
edges={}
for i in range(xi):
    for j in range(yi):
        grid[i][j].x_axis=round(x_min + num*i,5)
        grid[i][j].y_axis=round(y_min + num*j,5)
        x_1=grid[i][j].x_axis
        y_1=grid[i][j].y_axis
        x_2=round(x_1+num,5)
        y_2=round(y_1+num,5)
        if x_2>=x_max:
            x_2=x_max
        if y_2>=y_max:
            y_2=y_max
        print(x_2)
        print(y_2)
        add_edge(x_1,x_2,y_1,y_2,grid[i][j])
G=nx.Graph()
dirname = os.path.dirname(__file__)
filename = dirname + '/shape/crime_dt.shp'
data = shapefile.Reader(filename)
for i in range(len(data)):
    shape=data.shape(i)
    point=shape.points[0]
    x=point[0]
    y=point[1]
    x_grid=(x-x_min)/num
    y_grid=(y-y_min)/num
    if x_grid>=xi:
        x_grid=xi-1
    if y_grid>=yi:
        y_grid=yi-1
    grid[int(x_grid)][int(y_grid)].total += 1
all=0
grid_dict={}
for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j].total >threshold:
            grid[i][j].type=1
            grid_dict[grid[i][j].getKey()]=grid[i][j]
        else:
            grid[i][j].type=0
            grid_dict[grid[i][j].getKey()] = grid[i][j]
for key in edges.keys():
    if edges[key].isEdge==True:
        edges[key].cost=1000000
    else:
        g1=grid_dict[edges[key].g1.getKey()]
        g2=grid_dict[edges[key].g2.getKey()]
        if g1.getKey()==g2.getKey():
            if g1.type==1:
                edges[key].cost=1000000
            else:
                edges[key].cost=1.5
        else:
            if g1.type==g2.type==1:
                edges[key].cost=1000000
            elif g1.type==g2.type==0 :
                edges[key].cost=1.0
            else:
                edges[key].cost=1.3
def dest(a,b):
    (x1,y1)=a
    (x2,y2)=b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

for x in edges.values():
    G.add_edge((x.x1,x.y1),(x.x2,x.y2),weight=x.cost)
path=astar_path(G,(-73.55, 45.49),(-73.59, 45.53),dest)

plt.xticks(np.arange(x_min,x_max,num))
plt.yticks(np.arange(y_min,y_max,num))
plt.grid(True)
plt.tick_params(which='minor')
ax = plt.gca()
for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j].type==1:
            ax.add_patch(Rectangle(xy=(grid[i][j].x_axis,grid[i][j].y_axis),width=num,height=num,color='yellow', fill=True))
        else:
            ax.add_patch(Rectangle(xy=(grid[i][j].x_axis,grid[i][j].y_axis),width=num,height=num,color='black', fill=True))

if path !=None:
    for i in range(len(path) - 1):
        plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'ro-', linewidth=1.0, color='red')
plt.plot()
plt.show()
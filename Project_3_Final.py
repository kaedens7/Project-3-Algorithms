# -*- coding: utf-8 -*-
"""/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
Created on Sat Apr 09 00:01:14 2022                  /*
                                                     /*
CSCI 3330                                            /*
Algorithms                                           /*
Project 3                                            /*   
                                                     /*
Authors:                                             /*
    Brian Cavin                                      /*
    Landon Johnson                                   /*
    Kal Young                                        /*
                                                     /*   
/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*"""
import collections
from collections import defaultdict
from heapq import *
import sys
from itertools import groupby



'''************************************************

DFS algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''
# http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/

# The DFS function is a recursive implementation of
# the Depth-First-Search algorithm
# 
# -GRAPH: a dictionary of vertex and sdjancent list
# -START: starting vertex for traversal
# -VISITED: a set of visited vertices 
#

def dfs(graph, start, visited=None):


    # Initialization with empty set
    if visited is None:
        visited = set() 

    # Mark start visited and add it to visited
    visited.add(start)

    # For key in adjancency list set of start but
    # not yet visited visit the key
 
    for key in graph[start] - visited: # Python suports set subtraction       
        dfs(graph, key, visited) # DFS recursive call
    return visited

'''************************************************

BFS algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''
def bfs(graph, start):
    visited, queue = set(), [start]
    p =[]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            p.append(vertex)
            queue.extend(graph[vertex] - visited)
    return p

'''************************************************

BFS path algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''
# Reference code by UTD Xue Kris Yu

def dfs_path(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == goal:
                return path
            visited.add(vertex)
            for neighbor in graph[vertex]:
                stack.append((neighbor, path + [neighbor]))


'''************************************************

BFS path algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''

''' The modified script is contributed by Maarten Fabre available at
https://codereview.stackexchange.com/questions/193410/breadth-first-search-implementation-in-python-3-to-find-path-between-two-given-n
'''

def bfs_path(graph, start, goal):
    """
    finds a shortest path in undirected `graph` between `start` and `goal`. 
    If no path is found, returns `None`
    """
    if start == goal:
        return [start]
    visited = {start}
    queue = [(start, [])]

    while queue:
        current, path = queue.pop(0) 
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor == goal:
                return path + [current, neighbor]
            if neighbor in visited:
                continue
            queue.append((neighbor, path + [current]))
            visited.add(neighbor)   
    return None 

'''************************************************

SCC algoritm found on github by Landon Johnson
Modified for Project 3

************************************************'''

#https://github.com/ChuntaoLu/Algorithms-Design-and-
#Analysis/blob/master/week4%20Graph%20search%20and%20SCC/scc.py#L107

#The original script was written in Python 2. It computes the strong
# connected components(SCC) of a given graph.




#set rescursion limit and stack size limit
sys.setrecursionlimit(10 ** 6)
#resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, 2 ** 30))


class Tracker(object):
    """Keeps track of the current time, current source, component leader,
    finish time of each node and the explored nodes.
    
    'self.leader' is informs of {node: leader, ...}."""

    def __init__(self):
        self.current_time = 0
        self.current_source = None
        self.leader = {}
        self.finish_time = {}
        self.explored = set()


def scc_dfs(graph_dict, node, tracker):
    """Inner loop explores all nodes in a SCC. Graph represented as a dict,
    {tail: [head_list], ...}. Depth first search runs recursively and keeps
    track of the parameters"""

    tracker.explored.add(node)
    tracker.leader[node] = tracker.current_source
    for head in graph_dict[node]:
        if head not in tracker.explored:
            scc_dfs(graph_dict, head, tracker)
    tracker.current_time += 1
    tracker.finish_time[node] = tracker.current_time


def dfs_loop(graph_dict, nodes, tracker):
    """Outer loop checks out all SCCs. Current source node changes when one
    SCC inner loop finishes."""

    for node in nodes:
        if node not in tracker.explored:
            tracker.current_source = node
            scc_dfs(graph_dict, node, tracker)


def graph_reverse(graph):
    """Given a directed graph in forms of {tail:[head_list], ...}, compute
    a reversed directed graph, in which every edge changes direction."""

    reversed_graph = defaultdict(list)
    for tail, head_list in graph.items():
        for head in head_list:
            reversed_graph[head].append(tail)
    return reversed_graph


def scc(graph):
    """First runs dfs_loop on reversed graph with nodes in decreasing order,
    then runs dfs_loop on original graph with nodes in decreasing finish
    time order(obtained from first run). Return a dict of {leader: SCC}."""

    out = defaultdict(list)
    tracker1 = Tracker()
    tracker2 = Tracker()
    nodes = set()
    reversed_graph = graph_reverse(graph)
    for tail, head_list in graph.items():
        nodes |= set(head_list)
        nodes.add(tail)
    nodes = sorted(list(nodes), reverse=True)
    dfs_loop(reversed_graph, nodes, tracker1)
    sorted_nodes = sorted(tracker1.finish_time,
                          key=tracker1.finish_time.get, reverse=True)
    dfs_loop(graph, sorted_nodes, tracker2)
    for lead, vertex in groupby(sorted(tracker2.leader, key=tracker2.leader.get),
                                key=tracker2.leader.get):
        out[lead] = list(vertex)
    return out

'''************************************************

Dijkstra's algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''

# Python implementation of Dijkstra's algorithm
#https://gist.github.com/econchick/4666413

class Dijkstra_Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = collections.defaultdict(list)
    self.distances = {}

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance
                                        
def dijkstra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distance[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path

'''************************************************

Kruskal's algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''

# Python program for Kruskal's algorithm to find Minimum Spanning Tree
# of a given connected, undirected and weighted graph

#http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/
#This code is contributed by Neelam Yadav 


#Class to represent a graph
class Kruskal_Graph:
 
    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = [] # default dictionary to store graph
         
  
    # function to add an edge to graph
    def addEdge(self,u,v,w):
        self.graph.append([u,v,w])
 
    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        # Attach smaller rank tree under root of high rank tree
        # (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        #If ranks are same, then make one as root and increment
        # its rank by one
        else :
            parent[yroot] = xroot
            rank[xroot] += 1
 
    # The main function to construct MST using Kruskal's algorithm
    def KruskalMST(self):
 
        result =[] #This will store the resultant MST
 
        i = 0 # An index variable, used for sorted edges
        e = 0 # An index variable, used for result[]
 
        #Step 1:  Sort all the edges in non-decreasing order of their
        # weight.  If we are not allowed to change the given graph, we
        # can create a copy of graph
        self.graph =  sorted(self.graph,key=lambda item: item[2])
        #print self.graph
 
        parent = [] ; rank = []
 
        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
     
        # Number of edges to be taken is equal to V-1
        while e < self.V -1 :
 
            # Step 2: Pick the smallest edge and increment the index
            # for next iteration
            u,v,w =  self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent ,v)
 
            # If including this edge does't cause cycle, include it
            # in result and increment the index of result for next edge
            if x != y:
                e = e + 1  
                result.append([u,v,w])
                self.union(parent, rank, x, y)          
            # Else discard the edge
 
        # print the contents of result[] to display the built MST
        print ("Following are the edges in the constructed MST")
        for u,v,weight  in result:
            #print str(u) + " -- " + str(v) + " == " + str(weight)
            print ("%d -- %d == %d" % (u,v,weight))
 


 


'''************************************************

Prim's algoritm code provided by Dr. Chenyi Hu
Modified for Project 3

************************************************'''

#https://programmingpraxis.com/2010/04/09/minimum-spanning-tree-prims-algorithm/


 
def prim( nodes, edges ):
    conn = defaultdict( list )
    for n1,n2,c in edges:
        conn[ n1 ].append( (c, n1, n2) )
        conn[ n2 ].append( (c, n2, n1) )
 
    mst = []
    used = set( [nodes[ 0 ]] )
    usable_edges = conn[ nodes[0] ][:]
    heapify( usable_edges )
 
    while usable_edges:
        cost, n1, n2 = heappop( usable_edges )
        if n2 not in used:
            used.add( n2 )
            mst.append( ( n1, n2, cost ) )
 
            for e in conn[ n2 ]:
                if e[ 2 ] not in used:
                    heappush( usable_edges, e )
    return mst
 





def main():
    
    """/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
       Question 1:                                       /*
       Kal Young                                        /*   
    /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*"""
    
    '''********
    Question 1
    Part a 
    ********'''
   
    # Build undirected graph
    undirected_graph = {'A': set(['B', 'E', 'F']),
                        'B': set(['A', 'C', 'F']),
                        'C': set(['B', 'D', 'G']),
                        'D': set(['C', 'G']),
                        'E': set(['A', 'F', 'I']),
                        'F': set(['A', 'B', 'E', 'I']),
                        'G': set(['C', 'D', 'J']),
                        'H': set(['K', 'L']),
                        'I': set(['E','F', 'J','M']),
                        'J': set(['G', 'I']),
                        'K': set(['H', 'L', 'O']),
                        'L': set(['H', 'K', 'P']),
                        'M': set(['I', 'N']),
                        'N': set(['M']),
                        'O': set(['K']),
                        'P': set(['L'])}
    
    # Print graph
    print("Undirected graph:")  
    print(undirected_graph)
    print()
    
    print("Question 1")
    print("Part a\n")
    
    # DFS from different nodes
    print("Start DFS from node A:")
    undirected_dfs_graph = dfs(undirected_graph, 'A')
    print(undirected_dfs_graph)
    print()
    print("Start DFS from node P:")
    undirected_dfs_graph = dfs(undirected_graph, 'P')
    print(undirected_dfs_graph)
    print()
    
    # BFS from different nodes
    print("Start BFS from node A:")
    undirected_bfs_graph = bfs(undirected_graph, 'A')
    print(undirected_bfs_graph)
    print()
    print("Start DFS from node P:")
    undirected_bfs_graph = bfs(undirected_graph, 'P')
    print(undirected_bfs_graph)
    print()
    
    
    '''********
    Question 1
    Part b and c
    ********'''
    
    print("Question 1")
    print("Part b and c\n")
    
    # Perform DFS and BFS search from node A to N and compare results
    print("DFS path from A to N:")
    path = dfs_path(undirected_graph, 'A', 'N')
    
    if path:
        print(path)
    else:
        print('no path found')
        
    print()
    print("BFS path from A to N:")
    path = bfs_path(undirected_graph, 'A', 'N')
    
    if path:
        print(path)
    else:
        print('no path found')
        
    print()
    
    # Perform DFS and BFS search from node H to P and compare results
    print("DFS path from H to P:")
    path = dfs_path(undirected_graph, 'H', 'P')
    
    if path:
        print(path)
    else:
        print('no path found')
        
    print()
    print("BFS path from H to P:")
    path = bfs_path(undirected_graph, 'H', 'P')
    
    if path:
        print(path)
    else:
        print('no path found')
        
    print()
    
    # Perform DFS and BFS search from node A to P 
    # to check that search will return "no path found"
    print("DFS path from A to P:")
    path = dfs_path(undirected_graph, 'A', 'P')
    
    if path:
        print(path)
    else:
        print('no path found')
        
    print()
    print("BFS path from A to P:")
    path = bfs_path(undirected_graph, 'A', 'P')
    
    if path:
        print(path)
    else:
        print('no path found')
    print()
    
    
    """/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
       Question 2                                       /*
       Landon Johnson                                   /*   
    /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*"""
    
    
    '''********
    Question 2
    Part a
    ********'''
    print("Question 2")
    print("Part a\n")
    
    # Build graph
    graph = {
         '1': set(['3']),
         '2': set(['1']),
         '3': set(['2', '5']),
         '4': set(['1', '2', '12']),
         '5': set(['8', '6']),
         '6': set(['8', '7', '10']),
         '7': set(['10']),
         '8': set(['9', '10']),
         '9': set(['5', '11']),
         '10': set(['11']),
         '11': set(['12']),
         '12': set()
            }
   
    groups = scc(graph)
    
    #top_5 = heapq.nlargest(5, groups, key=lambda x: len(groups[x]))
    #sorted_groups = sorted(groups, key=lambda x: len(groups[x]), reverse=True)
    result = []
    for i in range(5):
        try:
            result.append(len(groups[top_5[i]]))
            #result.append(len(groups[sorted_groups[i]]))
        except:
            result.append(0)
    count = result
    components = groups
    
    
    print('Strongly connected components are:')
    for key in components:
        print(components[key])
    
    print()
    
    '''********
    Question 3
    Part a
   ********'''
        
    '''************************************************
    Run Dijkstra's Algorithm to determine shortest path tree
    ************************************************'''
    print("Question 3")
    print("Part a\n")
    dijkstra_graph = Dijkstra_Graph()        
    # Add nodes
    dijkstra_graph.nodes = {'A','B','C','D','E', 'F', 'G', 'H', 'I'}
    
    # Add edges
    dijkstra_graph.edges = {'A': ['B', 'C', 'D'], 
                            'B':['A', 'C', 'F', 'H'], 
                            'C':['A', 'B', 'D', 'E', 'F'],
                            'D':['A', 'C', 'E', 'I'],
                            'E':['C', 'D', 'F', 'G'],
                            'F':['B', 'C', 'E', 'G', 'H'],
                            'G':['E', 'F', 'H', 'I'],
                            'H':['B', 'F', 'G', 'I'],
                            'I':['D', 'G', 'H']}
    
    # Add edge weight
    dijkstra_graph.distance = {('A', 'B'):22, ('A', 'C'):9, ('A', 'D'):12,
                               ('B', 'A'):22, ('B', 'C'):35, ('B', 'F'):36, ('B', 'H'):34,
                               ('C', 'A'):9, ('C', 'B'):35, ('C', 'D'):4, ('C', 'F'):42, ('C', 'E'):65,
                               ('D', 'A'):12, ('D', 'C'):4, ('D', 'E'):33, ('D', 'I'):30, 
                               ('E', 'C'):65, ('E', 'D'):33, ('E', 'F'):18, ('E', 'G'):22, 
                               ('F', 'B'):36, ('F', 'C'):42, ('F', 'E'):18, ('F', 'G'):39, ('F', 'H'):24,   
                               ('G', 'E'):23, ('G', 'F'):39, ('G', 'H'):25, ('G', 'I'):21, 
                               ('H', 'B'):34, ('H', 'F'):24, ('H', 'G'):25, ('H', 'I'):19, 
                               ('I', 'D'):30, ('I', 'G'):21, ('I', 'H'):19}
      
    
    
    
    # Run algorithm from node 'A'
    v, path = dijkstra(dijkstra_graph, 'A')
    print("Dijkstra's Shortest path:")
    print('Visited: ', v)
    print('Path :', path)
    print("\n")
    
    
    
    """/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
       Question 3:                                       /*
       Brian Cavin                                       /*   
    /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*"""
    
    '''********
    Question 3
    Part b and c
    ********'''
    
    print("Question 3")
    print("Part b and c\n")
    
    '''************************************************
    Run Kruskal's Algorithm to compare to Prim MST results
    ************************************************'''
    
    # Node letters converted to numbers for this algorithm
    '''
    Key:
        A = 0
        B = 1
        C = 2
        D = 3
        E = 4
        F = 5
        G = 6
        H = 7
        I = 8
    '''
    print("Kruskal's MST:")
    kruskal_graph = Kruskal_Graph(9)
    kruskal_graph.addEdge(0, 1, 22)
    kruskal_graph.addEdge(0, 2, 9)
    kruskal_graph.addEdge(0, 3, 12)
    kruskal_graph.addEdge(1, 2, 35)
    kruskal_graph.addEdge(1, 5, 36)
    kruskal_graph.addEdge(1, 7, 34)
    kruskal_graph.addEdge(2, 3, 4)
    kruskal_graph.addEdge(2, 4, 65)
    kruskal_graph.addEdge(2, 5, 42)
    kruskal_graph.addEdge(3, 4, 33)
    kruskal_graph.addEdge(3, 8, 30)
    kruskal_graph.addEdge(4, 5, 18)
    kruskal_graph.addEdge(4, 6, 23)
    kruskal_graph.addEdge(5, 6, 39)
    kruskal_graph.addEdge(5, 7, 24)
    kruskal_graph.addEdge(6, 7, 25)
    kruskal_graph.addEdge(6, 8, 20)
    kruskal_graph.addEdge(7, 8, 19)


    kruskal_graph.KruskalMST()
    
    
    
    '''************************************************
    Run Prim's Algorithm to compare to Kruskal MST results
    ************************************************'''
    # Add nodes
    prim_nodes = list("ABCDEFGHI")
    
    # Add eddges and edge weights
    prim_edges = [ ('A', 'B', 22), ('A', 'C', 9), ('A', 'D', 12), 
              ('B', 'C', 35), ('B', 'F', 36), ('B', 'H', 34), 
              ('C', 'D', 4), ('C', 'E', 65), ('C', 'F', 42), 
              ('D', 'E', 33), ('D', 'I', 30), 
              ('E', 'F', 18), ('E', 'G', 23), 
              ('F', 'G', 39), ('F', 'H', 24), 
              ('G', 'H', 25), ('G', 'I', 21), 
              ('H', 'I', 19)]

    print("\n") 
    print ("Prim's MST:")
    print ( prim( prim_nodes, prim_edges ))
    
if __name__ == "__main__":
    main()

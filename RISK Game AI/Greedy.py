import heapq


class Vertex:
    def __init__(self, vertex, armies, ownValue):
        self.name = vertex
        self.neighbors = []
        self.armies = armies
        self.own = ownValue

    def add_neighbor(self, neighbor):
        if isinstance(neighbor, Vertex):
            if neighbor.name not in self.neighbors:
                self.neighbors.append(neighbor.name)
                neighbor.neighbors.append(self.name)
                self.neighbors = sorted(self.neighbors)
                neighbor.neighbors = sorted(neighbor.neighbors)
        else:
            return False

    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            if isinstance(neighbor, Vertex):
                if neighbor.name not in self.neighbors:
                    self.neighbors.append(neighbor.name)
                    neighbor.neighbors.append(self.name)
                    self.neighbors = sorted(self.neighbors)
                    neighbor.neighbors = sorted(neighbor.neighbors)
            else:
                return False


####################################################################################


class Graph:
    def __init__(self):
        self.vertices = {}
        self.neighbors = {}

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex):
            self.neighbors[vertex.name] = vertex.neighbors
            self.vertices[vertex.name] = vertex

    def getVertexByName(self, vertexname):
        return self.vertices[vertexname]

    def add_vertices(self, vertices):
        for vertex in vertices:
            if isinstance(vertex, Vertex):
                self.neighbors[vertex.name] = vertex.neighbors
                self.vertices[vertex.name] = vertex

    def add_edge(self, vertex_from, vertex_to):
        if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
            vertex_from.add_neighbor(vertex_to)
            if isinstance(vertex_from, Vertex) and isinstance(vertex_to, Vertex):
                self.neighbors[vertex_from.name] = vertex_from.neighbors
                self.neighbors[vertex_to.name] = vertex_to.neighbors

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def adjacencyList(self):
        if len(self.vertices) >= 1:
            return [str(key) + ":" + str(self.neighbors[key]) for key in self.neighbors.keys()]
        else:
            return dict()

    def adjacencyMatrix(self):
        if len(self.neighbors) >= 1:
            self.vertex_names = sorted(g.neighbors.keys())
            self.vertex_indices = dict(zip(self.vertex_names, range(len(self.vertex_names))))
            import numpy as np
            self.adjacency_matrix = np.zeros(shape=(len(self.neighbors), len(self.neighbors)))
            for i in range(len(self.vertex_names)):
                for j in range(i, len(self.neighbors)):
                    for el in g.neighbors[self.vertex_names[i]]:
                        j = g.vertex_indices[el]
                        self.adjacency_matrix[i, j] = 1
            return self.adjacency_matrix
        else:
            return dict()

    def getMatrixNeighbours(self, i):
        self.neighbours = []
        for j in range(len(self.adjacencyMatrix()[0])):
            a = int(self.adjacencyMatrix()[i][int(j)])
            self.neighbours.append(a)
        return self.neighbours


##################################################################################


def graph(g1):
    """ Function to print a graph as adjacency list and adjacency matrix. """
    return str(g1.adjacencyList()) + '\n' + '\n' + str(g1.adjacencyMatrix())


###################################################################################
h = []
gameA = []
gameOwn = []
attacking_list = []
target_list = []
turns = 0
vis = []


###################################################################################


###################################################################################
def numNeighbours(graph6, index, own7):
    num = 0
    neighbours2 = graph6.getMatrixNeighbours(index)
    for ii in range(len(neighbours2)):
        if neighbours2[ii] == 1 and own7[ii] == 0:
            num = num + 1
    return num


###################################################################################


def add_bonus(graph7, A3, own3, bonus):
    maximum = -1
    index111 = 0
    for y in range(len(A3)):
        if own3[y] == 1:
            numNei = numNeighbours(graph7, y, own3)
            if numNei > maximum:
                maximum = numNei
                index111 = y
    A3[index111] += bonus
    return index111


###################################################################################


def add_enemy_bonus(A4, own4):
    minimum = 1000000
    index222 = 0
    for n in range(len(A4)):
        if own4[n] == 0 and A4[n] < minimum:
            minimum = A4[n]
            index222 = n
    A4[index222] += 2
    return index222


###################################################################################


def inHeap(myHeap, enemy, my):
    for l in range(len(myHeap)):
        if myHeap[l][1] == enemy and myHeap[l][2] == my:
            return 1
    return 0


###################################################################################

# get all possible attacks
def can_attack(graph2, index12, own2, A2, heuristics):
    global h
    neighbours = graph2.getMatrixNeighbours(index12)
    for i in range(len(neighbours)):
        if own2[i] == 0 and neighbours[i] == 1 and own2[index12] == 1:
            if inHeap(h, i, A2[i]) == 0:
                if A2[index12] - A2[i] > 1:
                    heapq.heappush(h, (A2[i], i, index12))
                # else:
                #     heapq.heappush(h, (heuristics, i, index12))
            else:
                for j in range(len(h)):
                    if h[j][1] == i and h[j][2] == index12:
                        if A2[index12] - A2[i] > 1:
                            te = h[j][0]
                            te = min(te, A2[i])
                            h[j] = list(h[j])
                            h[j][0] = te
                            h[j] = tuple(h[j])
                        # else:
                        # te = h[j][0]
                        # te = min(te, heuristics)
                        # h[j] = list(h[j])
                        # h[j][0] = te
                        # h[j] = tuple(h[j])

###############################################################
def initialise_continent_List(contintents):
    global vis
    for z in range(len(contintents)):
        vis.append(0)
###############################################################
def add_continent_bonus(graph10, continents, bonus, A10, own10):
    global vis
    for c in range(len(continents)):
        for cc in range(len(continents[c])):
            if own10[continents[c][cc]] == 0:
                break
        if cc == len(continents[c]) and vis[c] == 0:
            add_bonus(graph10, A10, own10, bonus[c])
            vis[c] = 1
# ######################Greedy Method##########################


def greedy(graph1, continent, numCountries, bonus):
    global h
    global gameA
    global gameOwn
    global attacking_list
    global target_list

    A1 = []
    own1 = []
    #initialise_continent_List()
    turnsTerminationCounter = 0
    for kk in range(numCountries):
        kk = kk + 1
        vert = graph1.getVertexByName(kk)
        A1.append(vert.armies)
        own1.append(vert.own)
    index = add_bonus(graph1, A1, own1, 2)
    # Change
    attacking_list.append(index)
    target_list.append(0)
    gameA.append(A1.copy())
    gameOwn.append(own1.copy())

    for k in range(len(A1)):
        can_attack(graph1, k, own1, A1, enemy_territories(own1))
    while True:
        if len(h) == 0:
            # change
            attacking_list.append(0)
            target_list.append(0)
            if turnsTerminationCounter == 10:
                print("Draw")
                print("A: ")
                for u in range(len(A1)):
                    print(A1[u])
                print("own: ")
                for u in range(len(A1)):
                    print(own1[u])
                return False
            turnsTerminationCounter = turnsTerminationCounter + 1
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            index = add_enemy_bonus(A1, own1)
            # Change
            attacking_list.append(index)
            target_list.append(0)
            # Change
            attacking_list.append(0)
            target_list.append(0)
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            # Added second time because passive agent
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            index = add_bonus(graph1, A1, own1, 2)
            # Change
            attacking_list.append(index)
            target_list.append(0)
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            for k in range(len(A1)):
                can_attack(graph1, k, own1, A1, enemy_territories(own1))
            continue
        else:
            turnsTerminationCounter = 0
            element = heapq.heappop(h)
            attacked = element[1]
            attacking = element[2]
            numNeiAttacked = numNeighbours(graph1, attacked, own1)
            numNeiAttacking = numNeighbours(graph1, attacking, own1)
            # change
            attacking_list.append(attacking)
            target_list.append(attacked)
            if numNeiAttacked > numNeiAttacking:
                A1[attacking] -= A1[attacked]
                A1[attacked] = A1[attacking] - 1
                A1[attacking] = 1
                #add_continent_bonus(graph1, partitions, bonus, A1, own1)
            else:
               A1[attacking] -= A1[attacked]
               A1[attacking] = A1[attacking] - 1
               A1[attacked] = 1
               #add_continent_bonus(graph1, partitions, bonus, A1, own1)
            own1[attacked] = 1
            # else:
            #     attacking_list.append(0)
            #     target_list.append(0)
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            if enemy_territories(own1) == 0:
                print("Greedy Agent win")
                print("A: ")
                for u in range(len(A1)):
                    print(A1[u])
                print("own: ")
                for u in range(len(A1)):
                    print(own1[u])
                return True
            # add_continent_bonus()
            index = add_enemy_bonus(A1, own1)
            # Change
            attacking_list.append(index)
            target_list.append(0)
            # Change
            attacking_list.append(0)
            target_list.append(0)
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            index = add_bonus(graph1, A1, own1, 2)
            # Change
            attacking_list.append(index)
            target_list.append(0)
            gameA.append(A1.copy())
            gameOwn.append(own1.copy())
            while len(h) != 0:
                heapq.heappop(h)
            for k in range(len(A1)):
                can_attack(graph1, k, own1, A1, enemy_territories(own1))
            global turns
            turns = turns + 1

    # return False


##############################################################


def enemy_territories(own4):
    counter = 0
    for m in range(len(own4)):
        if own4[m] == 0:
            counter = counter + 1
    return counter


##############################################################
import re
def get_risk_graph_alg(file_name):
    global countries_1
    global countries_2
    global troops_1
    global troops_2

    #G = pydot.Dot(graph_type='graph')
    g = Graph()
    partition_bonus = []
    partitions = {}
    file = open(file_name, "r")
    line_1 = file.readline()
    line_1 = re.split(' |\n',line_1)
    line_1.remove('')
    n_nodes = int(line_1[0])
    n_edges = int(line_1[1])
    n_partitions = int(line_1[2])
    start_index = 2
    for i in range(start_index, start_index + n_nodes):
        start_index = i+1
        line = file.readline()
        line = re.split(' |\n|  |\r|\t',line)
        node_name = line[0]
        pl = line[1]
        army = line[2]
        if(pl == '2'):
            pl = '0'
        node = Vertex(int(node_name), int(army), int(pl))
        g.add_vertex(node)
    for i in range(start_index, start_index + n_edges):
        start_index = i+1
        line = file.readline(i)
        line = re.split(' |\n|  |\r|\t',line)
        node1 = line[0]
        node2 = line[1]
        node1 = g.getVertexByName(int(node1))
        node2 = g.getVertexByName(int(node2))
        g.add_edge(node1, node2)
    for i in range(start_index, start_index + n_partitions):
        line = file.readline(i)
        line = re.split(' |\n',line)
        line.remove('')
        bonus_value = int(line[0])
        partition_bonus.append(bonus_value)
        del line[0]
        for node_name in line:
            partitions[int(node_name)] = i-start_index + 1
    file.close()
    return (g, partition_bonus, partitions, n_nodes, n_partitions)

##############################################################

g, partition_bonus, partitions, n_nodes, n_partitions = get_risk_graph_alg("input3.txt")


# a = Vertex(1, 1, 1)
# b = Vertex(2, 2, 0)
# c = Vertex(3, 3, 0)
# d = Vertex(4, 4, 0)
# e = Vertex(5, 5, 1)
#
# a.add_neighbors([b, c])
# b.add_neighbors([a, c])
# c.add_neighbors([b, d, a, e])
# d.add_neighbor(c)
# e.add_neighbors([c])
#
# g = Graph()
# g.add_vertex(a)
# g.add_vertex(b)
# g.add_vertex(c)
# g.add_vertex(d)
# g.add_vertex(e)
# # g.add_vertices([a, b, c, d, e])
# g.add_edge(b, d)
print(g.adjacencyMatrix())
# print()
# # print(g.getMatrixNeighbours(4))
#
# A = [1, 2, 3, 4, 5]
# own = [1, 0, 0, 0, 1]
# continents = [1, 1, 2, 3, 3]
# bonus1 = [10, 20, 30]
#
# aa = g.getVertexByName(1)
# print(aa)
# numCountries1 = 5
#
greedy(g, partitions, n_nodes, partition_bonus)

# print("new A: ")
# for i in range(len(A)):
#     print(A[i])
#
# print("new own: ")
# for i in range(len(own)):
#     print(own[i])

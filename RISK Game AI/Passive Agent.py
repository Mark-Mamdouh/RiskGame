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
turns = 0


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


def add_bonus(A3, own3):
    minimum = 1000000
    index222 = 0
    for n in range(len(A3)):
        if own3[n] == 1 and A3[n] < minimum:
            minimum = A3[n]
            index222 = n
    A3[index222] += 2
    return index222


###################################################################################


def inHeap(myHeap, enemy, my):
    for l in range(len(myHeap)):
        if myHeap[l][1] == enemy and myHeap[l][2] == my:
            return 1
    return 0


###################################################################################


def can_attack(graph2, index12, own2, A2):
    # get enemy's territory with minimum armies
    minimum2 = 10000000
    index2 = -1
    neighbours = graph2.getMatrixNeighbours(index12)
    for i in range(len(neighbours)):
        if neighbours[i] == 1 and A2[i] < minimum2 and A2[index12] - A2[i] > 1 and own2[i] == 0:
            minimum2 = A2[i]
            index2 = i
    return index2


###############################################################
def make_completed_continent_list(continents3, continentList):
    for jj in range(len(continents3)):
        if continents3[jj] not in continentList:
            continentList.append(continents3[jj])


###############################################################


###############################################################
def make_continent_list(continents3, continentList):
    for jj in range(len(continents3)):
        if continents3[jj] not in continentList:
            continentList.append(continents3[jj])


###############################################################

###############################################################
# def add_continent_bonus(graph9, A8, own8, attackedIndex, continentList1, completedList, bonusList):
#     for m in continentList1:
#         counter = 0
#         for mm in range(len(continentList1)):
#             if continentList1[mm] == m and own8[mm] == 1 and m not in completedList:
#                 counter = counter + 1
#         if counter == :
#             add_bonus(graph9, A8, own8, bonusList[m])
###############################################################


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
            pl = '1'
        else:
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
    return g, partition_bonus, partitions, n_nodes, n_partitions


# ##############################Pacifist Method##################################
def passive(graph1, continent, numCountries, bonus):

    global gameA
    global gameOwn
    global attacking_list
    global target_list

    A1 = []
    own1 = []
    for kk in range(numCountries):
        kk = kk + 1
        vert = graph1.getVertexByName(kk)
        A1.append(vert.armies)
        own1.append(vert.own)

    # A1 = [1, 2, 3, 4, 5]
    # own1 = [1, 0, 0, 0, 1]
    # continent = [1, 1, 2, 3, 3]
    # bonus = [10, 20, 30]

    add_bonus(A1, own1)

##############################################################


g, partition_bonus, partitions, n_nodes, n_partitions = get_risk_graph_alg("input.txt")

passive(g, partitions, n_nodes, partition_bonus)

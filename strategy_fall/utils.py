class Node:
    def __init__(self, context, node_type=None, outward_edge=None, inward_edge=None, node_id=None):
        self.context = context
        self.node_type = node_type
        self.node_id = node_id
        self.outward_edge = outward_edge if outward_edge is not None else []
        self.inward_edge = inward_edge if inward_edge is not None else []
        self.degree = len(self.outward_edge) + len(self.inward_edge)

    def add_outward_edge(self, edge):
        if edge not in self.outward_edge:
            self.outward_edge.append(edge) 
            self.degree += 1
        return edge
    
    def add_outward_edges_by_list(self, edge_list):
        for edge in edge_list:
            self.add_outward_edge(edge)
        return edge_list
    
    def add_inward_edges_by_list(self, edge_list):
        for edge in edge_list:
            self.add_inward_edge(edge)
        return edge_list
    
    def add_inward_edge(self, edge):
        if edge not in self.inward_edge:
            self.inward_edge.append(edge)
            self.degree += 1
        return edge
    
    def remove_outward_edge(self, edge):
        if edge in self.outward_edge:
            self.outward_edge.remove(edge)
            self.degree -= 1
        return edge
    
    def remove_inward_edge(self, edge):
        if edge in self.inward_edge:
            self.inward_edge.remove(edge)
            self.degree -= 1
        return edge
    
    def get_outward_edge(self):
        return self.outward_edge
    
    def get_inward_edge(self):
        return self.inward_edge
    

class Edge:
    def __init__(self, source, target, weight=0, edge_type=None):
        self.source = source
        self.target = target
        self.weight = weight
        self.edge_type = edge_type

    def change_source(self, new_source):
        self.source = new_source

    def change_target(self, new_target):
        self.target = new_target

    def change_weight(self, new_weight):
        self.weight = new_weight

    def change_edge_type(self, new_edge_type):
        self.edge_type = new_edge_type

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def get_weight(self):
        return self.weight

class Graph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
        return node
    
    def add_edge(self, edge):
        self.add_node(edge.source)
        self.add_node(edge.target)
        if edge not in self.edges:
            self.edges.append(edge)
            edge.source.add_outward_edge(edge)
            edge.target.add_inward_edge(edge)
        return edge
    
    def add_node_by_edge(self, edge):
        return self.add_edge(edge)
    
    def add_node_by_list(self, node_list):
        for node in node_list:
            self.add_node(node)
        return node_list
    
    def add_edge_by_list(self, edge_list):
        for edge in edge_list:
            self.add_edge(edge)
        return edge_list
    
    def remove_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)
            edge.source.remove_outward_edge(edge)
            edge.target.remove_inward_edge(edge)
        return edge
    
    def remove_node(self, node):
        if node in self.nodes:
            related_edges = []
            for edge in self.edges:
                if edge.source == node or edge.target == node:
                    related_edges.append(edge)
            for edge in related_edges:
                self.remove_edge(edge)
            self.nodes.remove(node)
        return node

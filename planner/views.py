from django.shortcuts import render
from django.http import JsonResponse
import osmnx as ox
import networkx as nx
import numpy as np
import random
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import io
import base64

def home(request):
    return render(request, 'C:/Users/adity/Desktop/Django/RoadNetworkPlanner/planner/templates/planner/home.html')

def generate_road_network(request):
    if request.method == 'POST':
        location = request.POST.get('location', 'Berkeley, California, USA')
        
        try:
            # Fetch data from OSM
            graph = ox.graph_from_place(location, network_type="drive")
            tags = {"building": True}
            buildings = ox.features.features_from_place(location, tags)
            obstacles = buildings.geometry.tolist()

            # Generate original plot with buildings
            original_fig, original_ax = ox.plot_graph(graph, show=False, close=False)
            buildings.plot(ax=original_ax, color="red", alpha=0.5)
            before_plot_url = save_plot_to_base64(original_fig)

            # Generate optimized plot
            best_road_network = genetic_algorithm(graph, obstacles)
            optimized_fig, optimized_ax = ox.plot_graph(graph, show=False, close=False)
            plot_optimized_roads(optimized_ax, best_road_network)
            buildings.plot(ax=optimized_ax, color="red", alpha=0.5)
            after_plot_url = save_plot_to_base64(optimized_fig)

            return JsonResponse({
                'before_plot_url': before_plot_url,
                'after_plot_url': after_plot_url
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

def genetic_algorithm(graph, obstacles, generations=20, population_size=30):
    population = [generate_random_chromosome(graph) for _ in range(population_size)]
    for _ in range(generations):
        # Fitness evaluation
        fitness_scores = [fitness(chromosome, graph, obstacles) for chromosome in population]
        
        # Selection
        selected_indices = np.argsort(fitness_scores)[-population_size//2:]
        selected_population = [population[i] for i in selected_indices]
        
        # Crossover
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            new_population.append(child)
        
        # Mutation
        for i in range(len(new_population)):
            if random.random() < 0.1:
                mutation_point = random.randint(0, len(new_population[i]) - 1)
                u, v = random.sample(list(graph.nodes), 2)
                x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
                x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
                new_population[i][mutation_point] = (x1, y1, x2, y2)
        
        population = new_population
    
    return max(population, key=lambda x: fitness(x, graph, obstacles))

def fitness(chromosome, graph, obstacles):
    score = 0
    total_length = 0
    for segment in chromosome:
        x1, y1, x2, y2 = segment
        length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        total_length += length
        
        # Obstacle collision check
        road_segment = LineString([(x1, y1), (x2, y2)])
        for obstacle in obstacles:
            if road_segment.intersects(obstacle):
                score -= 100
    
    return score - total_length

def generate_random_chromosome(graph, num_segments=10):
    nodes = list(graph.nodes)
    return [
        (graph.nodes[u]["x"], graph.nodes[u]["y"], 
        graph.nodes[v]["x"], graph.nodes[v]["y"])
        for u, v in [random.sample(nodes, 2) for _ in range(num_segments)]
    ]

def plot_optimized_roads(ax, road_network):
    for segment in road_network:
        ax.plot([segment[0], segment[2]], 
                [segment[1], segment[3]], 
                "b-", linewidth=2)

def save_plot_to_base64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
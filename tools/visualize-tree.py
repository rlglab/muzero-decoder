#!/usr/bin/env python
from graphviz import Digraph
import re
import sys
from datetime import datetime


def parse_tree_description(tree_description):
    tree_structure = {}
    for line in tree_description.strip().split("\n"):
        parts = re.match(r"(\d+)\[([^\]]*)\]\{([^\}]*)\}{([^\}]*)\}", line)
        if parts:
            node_id, children, comments, visual_info = parts.groups()
            node_id = int(node_id)
            children = [int(child) for child in children.split(",") if child]
            comments = [comment.strip() for comment in comments.split(",") if comment]
            visual_info = [info.strip() for info in visual_info.split(",") if info]  # format: ["color=red"]
            visual_info = {info.split("=")[0]: info.split("=")[1] for info in visual_info}  # format {"color": "red"}
            tree_structure[node_id] = {"children": children, "comments": comments, "visual_info": visual_info}
    return tree_structure


def add_nodes_and_edges(tree_structure, node_id, dot, parent=None):
    node_label = "\n".join(tree_structure[node_id]['comments'])
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    node_label = ansi_escape.sub('', node_label)
    dot.node(str(node_id), label=node_label, fillcolor=tree_structure[node_id]['visual_info']['color'], style='filled', fontname="Consolas")

    if parent is not None:
        dot.edge(str(parent), str(node_id))

    for child in tree_structure[node_id]['children']:
        add_nodes_and_edges(tree_structure, child, dot, node_id)


def visualize_tree(tree_structure, tree_name, dir_name):
    if dir_name is None:
        now = datetime.now()
        time = now.strftime("%Y%m%d_%H%M%S")
        dir_name = f"./tree_dump/{time}"
    format = "png"
    dot = Digraph(comment=f'The Tree Structure for {tree_name}', node_attr={'fontname': 'Consolas'})
    try:
        add_nodes_and_edges(tree_structure, 0, dot)
    except RecursionError as err:
        print(f"Tree {tree_name} visualization raised " + str(err.args[0]), file=sys.stderr)
    file_name = f'{dir_name}/tree_{tree_name}'
    dot.render(file_name, format=format, cleanup=True)
    print(f"Tree {tree_name} visualization saved as {file_name}.{format}", file=sys.stderr)


def plot_mcts(dir_name=None, mcts_dump=None):
    current_tree_name = None
    current_description = []
    if mcts_dump is not None:
        mcts_dump = mcts_dump.split("\n")

    while True:
        if mcts_dump is None:
            try:
                line = input()
            except EOFError:
                break
        else:
            if len(mcts_dump) == 0:
                break
            line = mcts_dump.pop(0)
        if line.startswith("tree"):
            if current_tree_name and current_description:
                current_tree_structure = parse_tree_description("\n".join(current_description))
                visualize_tree(current_tree_structure, current_tree_name, dir_name)
            current_tree_name = line.split()[1]
            current_description = []
        elif line.strip() == "":
            if current_tree_name and current_description:
                current_tree_structure = parse_tree_description("\n".join(current_description))
                visualize_tree(current_tree_structure, current_tree_name, dir_name)
            else:
                break
            current_tree_name = None
            current_description = []
        else:
            current_description.append(line)


if __name__ == "__main__":
    plot_mcts(sys.argv[1] if len(sys.argv) >= 2 else None)

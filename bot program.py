from collections import deque

# Frequently asked questions database
info_store = {
    "library timings": "Library is open from 9 AM to 9 PM.",
    "food court timings": "Food court is open from 8 AM to 9 PM.",
    "sports complex timings": "Sports complex is open from 6 AM to 8 AM and 4:30 PM to 7 PM."
}

# Campus map represented as adjacency list
location_graph = {
    "Main Gate": ["Admin Block"],
    "Admin Block": ["Library (inside Admin)", "Engineering Wing"],
    "Library (inside Admin)": [],
    "Engineering Wing": ["Food Plaza"],
    "Food Plaza": ["Football Ground", "Hostel A"],
    "Football Ground": [],
    "Hostel A": ["Sports Lane"],
    "Sports Lane": ["Cricket Field", "Basketball Arena", "Volleyball Court"],
    "Cricket Field": [],
    "Basketball Arena": [],
    "Volleyball Court": []
}

def find_shortest_route(map_data, start_point, end_point):
    """Breadth-first search to get shortest route"""
    seen = set()
    queue = deque([[start_point]])

    while queue:
        route = queue.popleft()
        current = route[-1]

        if current == end_point:
            return route

        if current not in seen:
            for adj in map_data.get(current, []):
                queue.append(route + [adj])
            seen.add(current)

    return None

def process_input(message):
    msg = message.lower()

    # Greeting check
    if msg in ["hi", "hello", "hey"]:
        return "Hi there! How can I help you around campus?"

    # FAQ matching
    for key in info_store:
        if key in msg:
            return info_store[key]

    # Look for campus locations
    found_places = [loc for loc in location_graph if loc.lower() in msg]

    if len(found_places) >= 2:
        route = find_shortest_route(location_graph, found_places[0], found_places[1])
        if route:
            return "Route found: " + " -> ".join(route)
        else:
            return "No direct route found between those locations."

    return "I couldn’t understand. Try asking about timings or directions."

if __name__ == "__main__":
    print("Campus Helper Active! (type 'exit' to quit)")
    while True:
        user_text = input("You: ")
        if user_text.lower() == "exit":
            print("Helper: Goodbye!")
            break
        reply = process_input(user_text)
        print("Helper:", reply)

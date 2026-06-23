import os
import json
import random

def generate_expanded_dataset():
    print("=== Expanding Dataset to 50+ Test Graphs ===")
    os.makedirs(os.path.join("data", "webnlg"), exist_ok=True)
    
    # Base templates
    categories = ["Astronaut", "University", "Food", "Airport", "Monument", "Company", "City"]
    
    subjects = {
        "Astronaut": ["Buzz Aldrin", "Neil Armstrong", "Yuri Gagarin", "Sally Ride", "Chris Hadfield", "Mae Jemison"],
        "University": ["Harvard University", "Stanford University", "MIT", "Oxford University", "Yale University", "Cambridge University"],
        "Food": ["Pizza", "Sushi", "Taco", "Croissant", "Curry", "Gelato"],
        "Airport": ["Heathrow Airport", "JFK Airport", "Haneda Airport", "CDG Airport", "LAX Airport", "Schiphol Airport"],
        "Monument": ["Eiffel Tower", "Statue of Liberty", "Taj Mahal", "Colosseum", "Great Wall", "Machu Picchu"],
        "Company": ["Microsoft", "Google", "Amazon", "Meta", "Netflix", "Tesla"],
        "City": ["London", "New York", "Tokyo", "Paris", "Los Angeles", "Amsterdam"]
    }
    
    predicates = {
        "Astronaut": ["birthDate", "birthPlace", "nationality", "occupation", "was_selected_by_NASA"],
        "University": ["city", "state", "established", "type"],
        "Food": ["origin", "mainIngredient", "type"],
        "Airport": ["city", "elevation", "hubFor"],
        "Monument": ["location", "state", "established"],
        "Company": ["foundedBy", "headquarters", "keyPeople"]
    }
    
    # We will generate 50 graphs
    dataset = []
    
    # Let's seed for reproducibility
    random.seed(42)
    
    for i in range(1, 51):
        graph_id = f"graph_expanded_{i:03d}"
        
        # Decide graph scale: mix of small (5-15), medium (16-35), and large (36-60)
        if i <= 20:
            scale = random.randint(5, 15)
        elif i <= 40:
            scale = random.randint(16, 35)
        else:
            scale = random.randint(36, 60)
            
        triples = []
        seen = set()
        
        # Select domains/categories to include
        num_categories = max(1, scale // 7)
        selected_cats = random.sample(categories, min(num_categories, len(categories)))
        
        # Bridge entity
        common_country = random.choice(["United States", "Turkey", "Spain", "United Kingdom", "France", "Japan"])
        
        for cat in selected_cats:
            sub = random.choice(subjects[cat])
            preds = predicates.get(cat, ["relatedTo"])
            
            for p in preds:
                # Value generation
                if p == "birthDate" or p == "established":
                    val = f"{random.randint(1800, 2010)}-01-01"
                elif p == "birthPlace" or p == "city" or p == "location" or p == "headquarters":
                    val = random.choice(subjects["City"])
                elif p == "nationality" or p == "origin":
                    val = common_country
                else:
                    val = f"Value_{cat}_{random.randint(1,10)}"
                    
                t = (sub, p, val)
                if t not in seen:
                    seen.add(t)
                    triples.append(list(t))
                    
        # Add some bridging links to ensure connectivity
        for cat in selected_cats:
            if cat != "City":
                sub = random.choice(subjects[cat])
                t = (sub, "countryOfOrigin", common_country)
                if t not in seen:
                    seen.add(t)
                    triples.append(list(t))
                    
        # Limit to the targeted scale
        triples = triples[:scale]
        
        dataset.append({
            "id": graph_id,
            "scale": len(triples),
            "triples": triples
        })
        
    output_path = os.path.join("data", "webnlg", "expanded_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(dataset)} expanded graphs in {output_path}")

if __name__ == "__main__":
    generate_expanded_dataset()

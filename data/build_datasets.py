import os
import json

def build_datasets():
    # Create directories
    os.makedirs(os.path.join("data", "webnlg"), exist_ok=True)
    os.path.join("data", "generated_large")
    os.makedirs(os.path.join("data", "generated_large"), exist_ok=True)
    
    # 1. Define Native WebNLG-style subgraphs (small scale: 5-8 triples each)
    astronaut_triples = [
        ["Alan_Bean", "birthDate", "1932-03-15"],
        ["Alan_Bean", "birthPlace", "Wheeler, Texas"],
        ["Alan_Bean", "nationality", "United States"],
        ["Alan_Bean", "occupation", "Test pilot"],
        ["Alan_Bean", "was_selected_by_NASA", "1963"],
        ["Wheeler, Texas", "country", "United States"],
        ["Wheeler, Texas", "timeZone", "Central Time Zone"]
    ]
    
    university_triples = [
        ["Auburn University", "city", "Auburn, Alabama"],
        ["Auburn University", "state", "Alabama"],
        ["Auburn University", "established", "1856"],
        ["Auburn University", "type", "Public university"],
        ["Auburn, Alabama", "country", "United States"],
        ["Auburn, Alabama", "county", "Lee County, Alabama"]
    ]
    
    food_triples = [
        ["Baklava", "origin", "Turkey"],
        ["Baklava", "mainIngredient", "Phyllo dough, nuts, honey"],
        ["Baklava", "type", "Dessert"],
        ["Turkey", "capital", "Ankara"],
        ["Turkey", "currency", "Turkish lira"]
    ]
    
    airport_triples = [
        ["Adolfo Suarez Airport", "city", "Madrid"],
        ["Adolfo Suarez Airport", "elevation", "609 meters"],
        ["Madrid", "country", "Spain"],
        ["Madrid", "leaderName", "Manuela Carmena"],
        ["Spain", "capital", "Madrid"],
        ["Spain", "currency", "Euro"]
    ]
    
    monument_triples = [
        ["Alamo Plaza", "location", "San Antonio, Texas"],
        ["Alamo Plaza", "state", "Texas"],
        ["San Antonio, Texas", "country", "United States"],
        ["San Antonio, Texas", "mayor", "Ron Nirenberg"]
    ]

    company_triples = [
        ["Apple Inc", "foundedBy", "Steve Jobs"],
        ["Apple Inc", "headquarters", "Cupertino, California"],
        ["Apple Inc", "keyPeople", "Tim Cook"],
        ["Cupertino, California", "country", "United States"],
        ["Steve Jobs", "birthPlace", "San Francisco"]
    ]

    city_triples = [
        ["Ankara", "country", "Turkey"],
        ["Ankara", "populationTotal", "5747325"],
        ["Turkey", "memberOf", "NATO"],
        ["United States", "memberOf", "NATO"],
        ["Spain", "memberOf", "NATO"]
    ]

    # Save Native WebNLG Small Tasks
    native_small = [
        {"id": "webnlg_astronaut", "category": "Astronaut", "triples": astronaut_triples},
        {"id": "webnlg_university", "category": "University", "triples": university_triples},
        {"id": "webnlg_food", "category": "Food", "triples": food_triples},
        {"id": "webnlg_airport", "category": "Airport", "triples": airport_triples},
        {"id": "webnlg_monument", "category": "Monument", "triples": monument_triples},
        {"id": "webnlg_company", "category": "Company", "triples": company_triples}
    ]
    
    with open(os.path.join("data", "webnlg", "native_small.json"), "w", encoding="utf-8") as f:
        json.dump(native_small, f, indent=2, ensure_ascii=False)
    print("Created data/webnlg/native_small.json")

    # 2. Build Synthetic Long Graphs (large scale: 40-70 triples)
    # We chain the individual subgraphs together using the bridging relations:
    # - Wheeler, Texas -> US
    # - Auburn, Alabama -> US
    # - Cupertino, California -> US
    # - San Antonio, Texas -> US
    # - US -> memberOf -> NATO
    # - Turkey -> memberOf -> NATO
    # - Spain -> memberOf -> NATO
    # - Ankara -> Turkey
    # - Baklava -> origin -> Turkey
    # - Adolfo Suarez Airport -> Madrid -> Spain
    
    large_triples = []
    large_triples.extend(astronaut_triples)   # 7
    large_triples.extend(university_triples)  # 6
    large_triples.extend(food_triples)        # 5
    large_triples.extend(airport_triples)     # 6
    large_triples.extend(monument_triples)    # 4
    large_triples.extend(company_triples)     # 5
    large_triples.extend(city_triples)        # 5
    
    # Let's add more dense context to reach ~50 connected triples
    extra_triples = [
        ["Steve Jobs", "founded", "NeXT"],
        ["Steve Jobs", "nationality", "United States"],
        ["Tim Cook", "education", "Auburn University"],  # Bridges Apple Inc to Auburn University!
        ["Madrid", "timezone", "Central European Time"],
        ["Spain", "language", "Spanish"],
        ["Ankara", "timezone", "Turkey Time Zone"],
        ["United States", "capital", "Washington D.C."],
        ["NATO", "headquarters", "Brussels, Belgium"],
        ["Brussels, Belgium", "country", "Belgium"]
    ]
    large_triples.extend(extra_triples) # 9 more
    
    # Remove duplicate triples if any
    unique_triples = []
    seen = set()
    for t in large_triples:
        t_tuple = (t[0].strip(), t[1].strip(), t[2].strip())
        if t_tuple not in seen:
            seen.add(t_tuple)
            unique_triples.append(t)
            
    synthetic_large = [
        {
            "id": "synthetic_large_001",
            "category": "Cross-Domain-Chained",
            "triple_count": len(unique_triples),
            "triples": unique_triples
        }
    ]
    
    with open(os.path.join("data", "generated_large", "synthetic_large.json"), "w", encoding="utf-8") as f:
        json.dump(synthetic_large, f, indent=2, ensure_ascii=False)
    print(f"Created data/generated_large/synthetic_large.json with {len(unique_triples)} triples.")

if __name__ == "__main__":
    build_datasets()

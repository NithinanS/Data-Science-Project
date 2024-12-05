import asyncio
import aiohttp
from geopy.geocoders import Nominatim
import requests
import pandas as pd

# Geocoding function using async
async def geocode_affiliation(session, affiliation):
    geolocator = Nominatim(user_agent="dblp_scraper")
    try:
        location = geolocator.geocode(affiliation)
        if location:
            return affiliation, location.latitude, location.longitude
        return affiliation, None, None
    except Exception as e:
        print(f"Error geocoding {affiliation}: {e}")
        return affiliation, None, None

# Async function to fetch DBLP data
async def fetch_dblp_data(query, num_results=100):
    base_url = "https://dblp.org/search/publ/api"
    params = {"q": query, "format": "json", "h": num_results}
    response = requests.get(base_url, params=params)
    return response.json()

# Process data with async geocoding
async def process_dblp_data(query, num_results=100):
    dblp_data = await fetch_dblp_data(query, num_results)
    papers = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for hit in dblp_data['result']['hits']['hit']:
            try:
                info = hit['info']
                title = info.get("title", "Unknown Title")
                authors_raw = info.get("authors", {}).get("author", [])
                authors = ", ".join([author.get("text", "Unknown Author") for author in authors_raw]) if isinstance(authors_raw, list) else "Unknown Author"
                venue = info.get("venue", "Unknown Venue")
                year = info.get("year", "Unknown Year")
                affiliation = venue if venue else "Unknown Affiliation"
                
                # Create geocoding task for affiliation
                task = geocode_affiliation(session, affiliation)
                tasks.append(task)
                
                papers.append({
                    "title": title,
                    "authors": authors,
                    "affiliation": affiliation,
                    "venue": venue,
                    "year": year,
                })
            except Exception as e:
                print(f"Error processing record: {e}")
        
        # Wait for all geocoding tasks to complete
        geocode_results = await asyncio.gather(*tasks)
        
        # Attach geocode results to papers
        for i, paper in enumerate(papers):
            affiliation, lat, lon = geocode_results[i]
            paper["latitude"] = lat
            paper["longitude"] = lon
        
    return papers

# Main async entry point
async def main():
    query = "machine learning"
    num_results = 1000
    papers = await process_dblp_data(query, num_results)
    
    # Save data to CSV
    df = pd.DataFrame(papers)
    df.to_csv("dblp_papers_with_locations.csv", index=False)
    print("Data saved to dblp_papers_with_locations.csv")

# Run the asynchronous function
asyncio.run(main())

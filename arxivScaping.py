import urllib.request
import xml.etree.ElementTree as ET
import json
import time

keywords = [
    "Computer",
    "Economics",
    "Electrical Engineering",
    "Mathematics",
    "Physics",
    "Biology",
    "Finance",
    "Statistics"
]

# URL to search ArXiv API with the search term (replace space with "+")
base_url = 'http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results=1000'

# List to hold all extracted data in JSON format
data_list = []

# Record the start time
start_time = time.time()

# Loop through each keyword and fetch data
for keyword in keywords:
    # Format the search query by replacing spaces with '+'
    url = base_url.format(keyword.replace(" ", "+"))

    # Make the API request
    response = urllib.request.urlopen(url)
    xml_data = response.read().decode('utf-8')

    # Parse the XML data
    root = ET.fromstring(xml_data)

    # Loop through all entries in the feed
    count = 0  # Initialize count for each keyword
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        # Extract title, summary, publication date, and link
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        link = entry.find('{http://www.w3.org/2005/Atom}link').get('href')

        # Extract authors and affiliations
        authors = []
        for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
            name = author.find('{http://www.w3.org/2005/Atom}name').text
            affiliation = author.find('{http://arxiv.org/schemas/atom}affiliation')
            affiliation_text = affiliation.text if affiliation is not None else None
            authors.append({"name": name, "affiliation": affiliation_text})

        # Extract categories
        categories = []
        for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
            term = category.get('term')
            categories.append(term)

        # Extract journal_ref
        journal_ref = entry.find('{http://arxiv.org/schemas/atom}journal_ref')
        journal_ref_text = journal_ref.text if journal_ref is not None else None

        # Create a dictionary for the current entry
        entry_data = {
            "title": title,
            "summary": summary,
            "published": published,
            "link": link,
            "authors": authors,
            "categories": categories,
            "journal_ref": journal_ref_text
        }

        # Add this entry data to the list
        count += 1
        print(f"Appending Data Entries = {count} for keyword: {keyword}")
        data_list.append(entry_data)

    # Add a 3-second delay between requests to comply with ArXiv's rate limit
    print(f"Waiting for 3 seconds before fetching next keyword.")
    time.sleep(3)

# Convert the list of entries to JSON format and print it
json_data = json.dumps(data_list, indent=4)

# You can save this data to a file as well
with open('arxiv_data.json', 'w') as json_file:
    json_file.write(json_data)

# Record the end time
end_time = time.time()

# Calculate and print the time taken
elapsed_time = end_time - start_time

# Print the JSON output
print(f"Total entries fetched: {len(data_list)}")
print(f"Script executed in {elapsed_time:.2f} seconds")

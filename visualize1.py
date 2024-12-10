import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from wordcloud import WordCloud
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from pyvis.network import Network
from pymongo import MongoClient
import ast

# MongoDB connection setup
MONGO_URI = "mongodb+srv://root:root@dspaper.q2lhg.mongodb.net/"
DATABASE_NAME = "DSDE"
COLLECTION_NAMES = ["data2018", "data2019", "data2020", "data2021", "data2022", "data2023"]

@st.cache_data
def load_data_from_mongodb():
    """Load data from MongoDB."""
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    
    # Fetch each year's data and store them as individual DataFrames
    df2018 = pd.DataFrame(list(db["data2018"].find()))
    df2019 = pd.DataFrame(list(db["data2019"].find()))
    df2020 = pd.DataFrame(list(db["data2020"].find()))
    df2021 = pd.DataFrame(list(db["data2021"].find()))
    df2022 = pd.DataFrame(list(db["data2022"].find()))
    df2023 = pd.DataFrame(list(db["data2023"].find()))
    
    # Process the 'keyword' column safely
    for df in [df2018, df2019, df2020, df2021, df2022, df2023]:
        if "keyword" in df.columns:
            df["keyword"] = df["keyword"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
    
    return df2018, df2019, df2020, df2021, df2022, df2023

# Load data from MongoDB
df2018, df2019, df2020, df2021, df2022, df2023 = load_data_from_mongodb()

# Combine all years into one DataFrame
df_all_years = pd.concat([df2018, df2019, df2020, df2021, df2022, df2023], ignore_index=True)

# Check if the 'keyword' column exists before applying transformation
if "keyword" in df_all_years.columns:
    df_all_years["keyword"] = df_all_years["keyword"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
else:
    # If 'keyword' column doesn't exist, create it with empty lists (or handle as needed)
    df_all_years["keyword"] = pd.Series([], dtype="object")

@st.cache_data
def detect_communities(edges_str: str):
    """Community detection using greedy modularity communities."""
    edges = ast.literal_eval(edges_str)  # Safely evaluate the edge string
    G = nx.Graph(edges)
    return list(nx.community.greedy_modularity_communities(G))


st.title("Data Visualization")

topic = st.sidebar.selectbox(
    "Select Topic",
    [
        "Overview",
        "Data Visualization",
        "Spatial Data Visualization",
        "Network Visualization",
    ],
)

if topic == "Overview":
    st.header("Overview of the Project")
    st.write(
        """
        Welcome to the Data Visualization Project! This project demonstrates various data visualization techniques and tools,
        including data visualization for different years, spatial data visualizations, and network graph representations.
        
        **Key Topics Covered:**
        - Data Visualization: Visual representation of the data for different years with a stacked bar chart and pie chart.
        - Spatial Data Visualization: Geographical representation of random data points on a map.
        - Network Visualization: Creation of a network graph to represent connections between nodes.
        
        **Technologies Used:**
        - Streamlit for creating interactive dashboards.
        - Plotly and Matplotlib for creating charts and graphs.
        - NetworkX for network graph visualization.
        - Geopandas for spatial data representation on maps.
        
        Navigate through the sidebar to explore different topics and their visualizations.
    """
    )

elif topic == "Data Visualization":
    year = st.sidebar.selectbox(
        "Select Year", ["All", "2018", "2019", "2020", "2021", "2022", "2023"]
    )

    if year == "All":
        st.write("Amount of data for each year")

        st.markdown(
            """
        <style>
        .year-box {
            background-color: #f9f9f9;
            border: 2px solid #0073e6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            transition: transform 0.2s, box-shadow 0.2s; /* Smooth hover animation */
        }
        .year-box:hover {
            transform: translateY(-10px); /* Move up slightly */
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); /* Enhance shadow */
            background-color: #e6f7ff; /* Light blue background on hover */
        }
        .year-header {
            font-size: 18px;
            font-weight: bold;
            color: #0073e6;
            margin-bottom: 10px;
        }
        .count {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        a1, a2, a3, a4, a5, a6 = st.columns(6)

        for col, df, year in zip(
            [a1, a2, a3, a4, a5, a6],
            [df2018, df2019, df2020, df2021, df2022, df2023],
            ["2018", "2019", "2020", "2021", "2022", "2023"],
        ):
            count = df.shape[0]
            with col:
                st.markdown(
                    f"""
                <div class="year-box">
                    <p class="year-header">Year {year}</p>
                    <p class="count">{count}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.write("---")

        st.header("Stacked Bar Chart with Dashboard")

    # Generate year_data_map dynamically
    year_data_map = {
        "2018": df2018,
        "2019": df2019,
        "2020": df2020,
        "2021": df2021,
        "2022": df2022,
        "2023": df2023,
    }

    # Multiselect for years
    years = st.multiselect(
        "Select Years",
        options=list(year_data_map.keys()),  # Use dynamic keys
        default=list(year_data_map.keys())  # Select all years by default
    )

    # Filter selected years and create aggregated data
    if years:
        aggregated_data = pd.concat([year_data_map[yr] for yr in years], ignore_index=True)

        if "subjectCode" in aggregated_data.columns:
            # Aggregate by subjectCode
            subject_counts = aggregated_data["subjectCode"].value_counts().reset_index()
            subject_counts.columns = ["Subject Code", "Count"]

            st.header(f"Histogram of Subject Codes for Selected Years")

            # Plot histogram
            fig = px.histogram(
                subject_counts,
                x="Subject Code",
                y="Count",
                labels={"Subject Code": "Subject", "Count": "Count"},
                title="Distribution of Subject Codes"
            )
            st.plotly_chart(fig)

            # Display top categories
            top_categories = subject_counts["Subject Code"].head(10)
            st.write("---")
            st.header(f"Trend of Top 10 Subjects for Selected Years")

            categories = st.multiselect(
                "Select Categories", options=top_categories, default=top_categories
            )

            if categories:
                filtered_data = aggregated_data[aggregated_data["subjectCode"].isin(categories)]

                # Count occurrences of the selected categories
                category_counts = (
                    filtered_data["subjectCode"].value_counts().reset_index()
                )
                category_counts.columns = ["Subject Code", "Count"]

                # Plot pie chart
                fig1 = px.pie(
                    category_counts,
                    names="Subject Code",
                    values="Count",
                    title="Distribution of Selected Categories",
                    labels={"Subject Code": "Subject", "Count": "Count"},
                )
                st.plotly_chart(fig1)
            else:
                st.warning("Please select at least one category.")
        else:
            st.error("The 'subjectCode' column is missing from the data.")
    else:
        st.warning("No years selected. Please select at least one year.")
elif topic == "Spatial Data Visualization":
    st.header("Spatial Data Visualization !! ยังไม่ทำ !!!")

    # spatial_data = {
    #     "Latitude": np.random.uniform(-90, 90, len(df_all_years)),
    #     "Longitude": np.random.uniform(-180, 180, len(df_all_years)),
    #     "Title": df_all_years["title"],
    #     "Year": df_all_years["year"],
    # }
    # spatial_df = pd.DataFrame(spatial_data)

    # st.subheader("Geographical Distribution of Publications")

    # geometry = [
    #     Point(xy) for xy in zip(spatial_df["Longitude"], spatial_df["Latitude"])
    # ]
    # gdf = gpd.GeoDataFrame(spatial_df, geometry=geometry)

    # fig = px.scatter_geo(
    #     gdf,
    #     lat="Latitude",
    #     lon="Longitude",
    #     text="Title",
    #     color="Year",
    #     title="Geographical Distribution of Publications (by Year)",
    # )
    # st.plotly_chart(fig)

elif topic == "Network Visualization":
    st.header("Network Visualization")

    # Sidebar for sample size selection
    sample_size = st.sidebar.slider(
        "Select the number of samples for network visualization",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
    )

    # Randomly sample data for visualization
    sample_data = df_all_years.sample(n=sample_size, random_state=42)

    # Generate edges for the graph from the 'keyword' column
    edges = []
    for keywords in sample_data["keyword"]:
        for i, k1 in enumerate(keywords[:-1]):
            for k2 in keywords[i + 1 :]:
                edges.append((k1, k2))

    # Initialize the graph using NetworkX
    G = nx.Graph()
    G.add_edges_from(edges)

    # Sidebar options for layout and centrality measure
    layout_option = st.sidebar.selectbox(
        "Select layout for visualization",
        ["spring", "kamada_kawai", "circular", "random"],
    )

    centrality_option = st.sidebar.selectbox(
        "Choose node size by centrality measure",
        ["degree", "betweenness", "closeness", "pagerank"],
    )

    # Sidebar sliders for customization
    node_size_range = st.sidebar.slider(
        "Node Size Range",
        min_value=5,
        max_value=200,
        value=(10, 50),
        step=5,
    )

    graph_size = st.sidebar.slider(
        "Graph Size",
        min_value=500,
        max_value=3000,
        value=1000,
        step=100,
    )

    # Adjust node spacing for spring layout
    if layout_option == "spring":
        node_spacing = st.sidebar.slider(
            "Node Spacing",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
        )
    else:
        node_spacing = 2.0

    # Font size and style for node labels
    font_size = st.sidebar.slider(
        "Label Font Size",
        min_value=8,
        max_value=40,
        value=16,
        step=2,
    )

    font_style = st.sidebar.selectbox(
        "Select Font Style",
        ["Arial", "Comic Sans MS", "Courier New", "Tahoma", "Times New Roman"],
        index=0,
    )

    # Option to show or hide edges
    show_edges = st.sidebar.checkbox("Show Edges", value=True)

    # Apply layout algorithm based on selection
    if layout_option == "spring":
        pos = nx.spring_layout(G, k=node_spacing / 10.0)
    elif layout_option == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_option == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G)

    # Apply centrality measure to determine node sizes
    if centrality_option == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_option == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif centrality_option == "closeness":
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.pagerank(G)

    # Node sizes based on centrality measure
    node_sizes = [centrality[node] * 1000 for node in G.nodes]

    # Detect communities if requested
    st.markdown("---")  # Add separator
    show_communities = st.checkbox("Detect Communities")
    communities = None
    community_stats_container = st.empty()  # Placeholder for community stats

    if show_communities:
        try:
            # Detect communities
            edges_str = str(list(G.edges()))
            communities_iter = detect_communities(edges_str)
            communities = {}

            # Create a list to store community sizes
            community_sizes = []

            for idx, community in enumerate(communities_iter):
                community_sizes.append(len(community))
                for node in community:
                    communities[node] = idx

            # Display community statistics
            with community_stats_container.container():
                st.caption("Community Statistics")
                st.metric("Number of Communities", len(communities_iter))
                avg_size = sum(community_sizes) / len(community_sizes)
                st.metric("Average Community Size", f"{avg_size:.1f}")

                # Sort communities by size
                community_df = pd.DataFrame(
                    {"Community": range(len(community_sizes)), "Size": community_sizes}
                ).sort_values("Size", ascending=False)

                st.caption("Community Sizes (sorted by size):")
                st.dataframe(
                    community_df,
                    hide_index=True,
                    height=min(len(community_sizes) * 35 + 38, 300),
                )

        except Exception as e:
            st.warning(f"Could not detect communities: {str(e)}")

    # Visualize the graph using matplotlib and networkx
    fig, ax = plt.subplots(figsize=(10, 8))  # Set the figure size
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_size=node_sizes,
        node_color="skyblue",
        font_size=font_size,
        font_weight="bold",
        edge_color="gray",
    )

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Word Cloud for keyword frequency
    st.subheader("Keyword Frequency Word Cloud")
    all_keywords = sum(sample_data["keyword"], [])
    word_freq = pd.Series(all_keywords).value_counts()
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    st.image(wordcloud.to_array(), use_container_width=True)

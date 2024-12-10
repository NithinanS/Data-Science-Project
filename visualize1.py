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

df2018 = pd.read_csv("FilteredDataWithYear/data2018.csv")
df2019 = pd.read_csv("FilteredDataWithYear/data2019.csv")
df2020 = pd.read_csv("FilteredDataWithYear/data2020.csv")
df2021 = pd.read_csv("FilteredDataWithYear/data2021.csv")
df2022 = pd.read_csv("FilteredDataWithYear/data2022.csv")
df2023 = pd.read_csv("FilteredDataWithYear/data2023.csv")

df_all_years = pd.concat([df2018, df2019, df2020, df2021, df2022, df2023])

df_all_years["keyword"] = df_all_years["keyword"].apply(
    lambda x: eval(x)
)

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

        for col, year, df in zip(
            [a1, a2, a3, a4, a5, a6],
            ["2018", "2019", "2020", "2021", "2022", "2023"],
            [df2018, df2019, df2020, df2021, df2022, df2023],
        ):
            count = df["title"].count()
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
        data2018 = df2018["subjectCode"].value_counts().reset_index()
        data2019 = df2019["subjectCode"].value_counts().reset_index()
        data2020 = df2020["subjectCode"].value_counts().reset_index()
        data2021 = df2021["subjectCode"].value_counts().reset_index()
        data2022 = df2022["subjectCode"].value_counts().reset_index()
        data2023 = df2023["subjectCode"].value_counts().reset_index()

        year_data_map = {
            "2018": data2018,
            "2019": data2019,
            "2020": data2020,
            "2021": data2021,
            "2022": data2022,
            "2023": data2023,
        }

        years = st.multiselect(
            "Select Years",
            options=["2018", "2019", "2020", "2021", "2022", "2023"],
            default=["2018", "2019", "2020", "2021", "2022", "2023"],
        )

    else:
        year_data_map = {
            '2018': df2018,
            '2019': df2019,
            '2020': df2020,
            '2021': df2021,
            '2022': df2022,
            '2023': df2023,
        }

        if year in year_data_map:
            data = year_data_map[year]

            subject_data = data['subjectCode'].value_counts().reset_index()
            subject_data.columns = ['Subject Code', 'Count']

            st.header(f'Histogram of Subject Codes for {year}')

            fig = px.histogram(
                subject_data, 
                x='Subject Code', 
                y='Count',
                labels={'Subject Code': 'Subject', 'Count': 'Count'}
            )
            st.plotly_chart(fig)

            top_categories = subject_data['Subject Code'].head(10)
            st.write('---')
            st.header(f"Trend of Top 10 Subjects for {year}")

            categories = st.multiselect(
                "Select Categories",
                options=top_categories,
                default=top_categories
            )

            if categories:
                filtered_data = data[data['subjectCode'].isin(categories)]

                # Count occurrences of the selected categories
                category_counts = filtered_data['subjectCode'].value_counts().reset_index()
                category_counts.columns = ['Subject Code', 'Count']

                # Plot a pie chart instead of a stacked bar chart
                fig1 = px.pie(
                    category_counts, 
                    names='Subject Code', 
                    values='Count', 
                    title=f"Distribution of Selected Subjects for {year}",
                    labels={'Subject Code': 'Subject', 'Count': 'Count'}
                )
                st.plotly_chart(fig1)

            else:
                st.write("Please select at least one category.")
        else:
            st.error("Selected year data is unavailable.")

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
    
    sample_size = st.sidebar.slider(
        "Select the number of samples for network visualization",
        min_value=5,
        max_value=50,
        value=20,  # default value
        step=5,
    )

    sample_data = df_all_years.sample(
        n=sample_size, random_state=42
    )  # random sample with a fixed seed for reproducibility

    edges = []
    for keywords in sample_data["keyword"]:
        for i, k1 in enumerate(keywords[:-1]):
            for k2 in keywords[i + 1 :]:
                edges.append((k1, k2))

    G = nx.Graph()
    G.add_edges_from(edges)

    layout_option = st.sidebar.selectbox(
        "Select layout for visualization",
        ["spring", "kamada_kawai", "circular", "random"],
    )

    centrality_option = st.sidebar.selectbox(
        "Choose node size by centrality measure",
        ["degree", "betweenness", "closeness", "pagerank"],
    )

    node_size_range = st.sidebar.slider(
        "Node Size Range",
        min_value=5,
        max_value=200,
        value=(10, 50),
        step=5,
        help="Set the minimum and maximum node sizes",
    )

    graph_size = st.sidebar.slider(
        "Graph Size",
        min_value=500,
        max_value=3000,
        value=1000,
        step=100,
        help="Adjust the overall size of the graph",
    )

    if layout_option == "spring":
        node_spacing = st.sidebar.slider(
            "Node Spacing",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Adjust the spacing between nodes (only for spring layout)",
        )
    else:
        node_spacing = 2.0

    font_size = st.sidebar.slider(
        "Label Font Size",
        min_value=8,
        max_value=40,
        value=16,
        step=2,
        help="Adjust the font size of node labels",
    )

    font_style = st.sidebar.selectbox(
        "Select Font Style",
        ["Arial", "Comic Sans MS", "Courier New", "Tahoma", "Times New Roman"],
        index=0,  # Default is Arial
        help="Choose a font style for the node labels",
    )

    show_edges = st.sidebar.checkbox(
        "Show Edges", value=True, help="Toggle visibility of edges"
    )

    if layout_option == "spring":
        pos = nx.spring_layout(G, k=node_spacing / 10.0)
    elif layout_option == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_option == "circular":
        pos = nx.circular_layout(G)
    else:  # Random layout
        pos = nx.random_layout(G)

    if centrality_option == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_option == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif centrality_option == "closeness":
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.pagerank(G)

    node_sizes = [centrality[node] * 1000 for node in G.nodes]

    fig, ax = plt.subplots(figsize=(8, 8))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=node_sizes,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
        width=2 if show_edges else 0,
        font_family=font_style,
    )

    st.pyplot(fig)

    st.subheader("Keyword Frequency Word Cloud")
    all_keywords = sum(sample_data["keyword"], [])
    word_freq = pd.Series(all_keywords).value_counts()
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    st.image(wordcloud.to_array(), use_column_width=True)

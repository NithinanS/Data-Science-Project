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
import plotly.graph_objects as go

df2018 = pd.read_csv("FilteredDataWithYear/data2018.csv")
df2019 = pd.read_csv("FilteredDataWithYear/data2019.csv")
df2020 = pd.read_csv("FilteredDataWithYear/data2020.csv")
df2021 = pd.read_csv("FilteredDataWithYear/data2021.csv")
df2022 = pd.read_csv("FilteredDataWithYear/data2022.csv")
df2023 = pd.read_csv("FilteredDataWithYear/data2023.csv")

df_all_years = pd.concat([df2018, df2019, df2020, df2021, df2022, df2023])

df_all_years["keyword"] = df_all_years["keyword"].apply(lambda x: eval(x))

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
            "2018": df2018,
            "2019": df2019,
            "2020": df2020,
            "2021": df2021,
            "2022": df2022,
            "2023": df2023,
        }

        if year in year_data_map:
            data = year_data_map[year]

            subject_data = data["subjectCode"].value_counts().reset_index()
            subject_data.columns = ["Subject Code", "Count"]

            st.header(f"Histogram of Subject Codes for {year}")

            fig = px.histogram(
                subject_data,
                x="Subject Code",
                y="Count",
                labels={"Subject Code": "Subject", "Count": "Count"},
            )
            st.plotly_chart(fig)

            top_categories = subject_data["Subject Code"].head(10)
            st.write("---")
            st.header(f"Trend of Top 10 Subjects for {year}")

            categories = st.multiselect(
                "Select Categories", options=top_categories, default=top_categories
            )

            if categories:
                filtered_data = data[data["subjectCode"].isin(categories)]

                # Count occurrences of the selected categories
                category_counts = (
                    filtered_data["subjectCode"].value_counts().reset_index()
                )
                category_counts.columns = ["Subject Code", "Count"]

                # Plot a pie chart instead of a stacked bar chart
                fig1 = px.pie(
                    category_counts,
                    names="Subject Code",
                    values="Count",
                    title=f"Distribution of Selected Subjects for {year}",
                    labels={"Subject Code": "Subject", "Count": "Count"},
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
    st.title("Network Visualization Tool")

    # Input: Network data
    with st.sidebar:
        st.subheader("Choose Network Data")
        data_source = st.radio(
            "Select data source",
            ["Sample Networks", "Graph Generator", "Upload Network"],
        )

        G = None  # Initialize graph

        # Sample Networks
        if data_source == "Sample Networks":
            sample_option = st.selectbox(
                "Select sample network", ["Karate Club", "Les Miserables"]
            )
            if sample_option == "Karate Club":
                G = nx.karate_club_graph()
            elif sample_option == "Les Miserables":
                G = nx.les_miserables_graph()

        # Graph Generator
        elif data_source == "Graph Generator":
            generator_type = st.selectbox(
                "Select generator type",
                [
                    "Complete Graph",
                    "Random (Erdős-Rényi)",
                    "Small World (Watts-Strogatz)",
                    "Scale-free (Barabási-Albert)",
                ],
            )

            if generator_type == "Complete Graph":
                n = st.slider("Number of nodes", 3, 100, 20)
                G = nx.complete_graph(n)

            elif generator_type == "Random (Erdős-Rényi)":
                n = st.slider("Number of nodes", 3, 100, 20)
                p = st.slider("Edge probability", 0.0, 1.0, 0.2)
                G = nx.erdos_renyi_graph(n, p)

            elif generator_type == "Small World (Watts-Strogatz)":
                n = st.slider("Number of nodes", 100, 1000, 200)
                k = st.slider("Number of nearest neighbors", 4, 20, 6)
                p = st.slider("Rewiring probability", 0.0, 1.0, 0.1)
                G = nx.watts_strogatz_graph(n, k, p)

            elif generator_type == "Scale-free (Barabási-Albert)":
                n = st.slider("Number of nodes", 100, 1000, 200)
                m = st.slider("Number of edges to attach", 1, 10, 3)
                G = nx.barabasi_albert_graph(n, m)

        # Upload Network
        elif data_source == "Upload Network":
            file_format = st.selectbox("Choose file format", ["CSV", "GML", "GraphML"])
            uploaded_file = st.file_uploader(
                "Upload network file", type=["csv", "gml", "graphml"]
            )

            if uploaded_file:
                G = load_network_from_file(uploaded_file, file_format)
                if G is None:
                    st.info("Please ensure your file is properly formatted")
                    st.stop()

            else:
                st.info("Please upload a network file or use other data sources")
                st.stop()

    if G is not None:
        # Visualization Options
        st.subheader("Visualization Options")

        # Layout selection
        layout_option = st.selectbox(
            "Layout Algorithm", ["spring", "kamada_kawai", "circular", "random"]
        )

        # Centrality metrics selection
        centrality_option = st.selectbox(
            "Node Size By", ["degree", "betweenness", "closeness", "pagerank"]
        )

        # Size controls
        scale_factor = st.slider(
            "Graph Size",
            min_value=500,
            max_value=3000,
            value=1000,
            step=100,
            help="Adjust the overall size of the graph",
        )

        if layout_option == "spring":
            node_spacing = st.slider(
                "Node Spacing",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                help="Adjust the spacing between nodes (only for spring layout)",
            )
        else:
            node_spacing = 2.0

        node_size_range = st.slider(
            "Node Size Range",
            min_value=5,
            max_value=200,
            value=(10, 50),
            step=5,
            help="Set the minimum and maximum node sizes",
        )

        # Node label font size
        font_size = st.slider(
            "Label Font Size",
            min_value=8,
            max_value=40,
            value=16,
            step=2,
            help="Adjust the font size of node labels",
        )

        # Edge visibility toggle
        show_edges = st.checkbox(
            "Show Edges", value=True, help="Toggle edge visibility"
        )

        # Community detection checkbox
        show_communities = st.checkbox("Detect Communities")

        # Initialize communities variable
        communities = None
        community_stats_container = st.empty()  # Create a placeholder for stats

        if show_communities:
            try:
                edges_str = str(list(G.edges()))
                communities_iter = detect_communities(edges_str)
                communities = {}

                # Create a list to store community sizes
                community_sizes = []

                for idx, community in enumerate(communities_iter):
                    community_sizes.append(len(community))
                    for node in community:
                        communities[node] = idx

                # Use the placeholder to display community statistics
                with community_stats_container.container():
                    st.caption("Community Statistics")
                    st.metric("Number of Communities", len(communities_iter))
                    avg_size = sum(community_sizes) / len(community_sizes)
                    st.metric("Average Community Size", f"{avg_size:.1f}")

                    # Sort communities by size in descending order
                    community_df = pd.DataFrame(
                        {
                            "Community": range(len(community_sizes)),
                            "Size": community_sizes,
                        }
                    ).sort_values("Size", ascending=False)

                    st.caption("Community Sizes (sorted by size):")
                    st.dataframe(
                        community_df,
                        hide_index=True,
                        height=min(len(community_sizes) * 35 + 38, 300),
                    )

            except Exception as e:
                st.warning(f"Could not detect communities: {str(e)}")

        # Main visualization area (now full width)
        if G is not None:
            # Initialize analyzers and visualizer
            analyzer = NetworkAnalyzer(G)
            visualizer = NetworkVisualizer(G)

            # Display basic stats in a more compact form
            st.text(
                f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())} | "
                f"Density: {nx.density(G):.3f}"
            )

            # Create and display visualization with increased height
            html_file = visualizer.create_interactive_network(
                communities=communities,
                layout=layout_option,
                centrality_metric=centrality_option,
                scale_factor=scale_factor,
                node_spacing=node_spacing,
                node_size_range=node_size_range,
                show_edges=show_edges,
                font_size=font_size,
            )

            st.markdown(
                f'<iframe src="{html_file}" width="100%" height="700px" frameborder="0"></iframe>',
                unsafe_allow_html=True,
            )

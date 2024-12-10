import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from wordcloud import WordCloud
import networkx as nx
from pyvis.network import Network
import geopandas as gpd
from shapely.geometry import Point

df2018 = pd.read_csv("data/Data2018.csv")
df2019 = pd.read_csv("data/Data2019.csv")
df2020 = pd.read_csv("data/Data2020.csv")
df2021 = pd.read_csv("data/Data2021.csv")
df2022 = pd.read_csv("data/Data2022.csv")
df2023 = pd.read_csv("data/Data2023.csv")

df_all_years = pd.concat([df2018, df2019, df2020, df2021, df2022, df2023])

df_all_years["keyword"] = df_all_years["keyword"].apply(lambda x: eval(x))



@st.cache_data  # Credit: Veera Muangsin
def detect_communities(edges_str: str):
    """Community detection using greedy modularity communities"""
    # Recreate graph from edges string
    edges = eval(edges_str)
    G = nx.Graph(edges)
    return list(nx.community.greedy_modularity_communities(G))

st.set_page_config(page_title="Data Visualization", layout="wide")
st.title("Data Visualization")

st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #444444;
    }
    .stSelectbox label {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-title">🗒️ Select Page</div>', 
    unsafe_allow_html=True
)

topic = st.sidebar.selectbox(
    "Select Topic",
    [
        "Overview",
        "Data Visualization",
        "Network Visualization",
    ],
    format_func=lambda x: (
        f"📄 {x}"
        if x == "Overview"
        else (
            f"📊 {x}"
            if x == "Data Visualization"
            else f"🔗 {x}" if x == "Network Visualization" else x
        )
    ),
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
        data2018["Year"] = "2018"
        data2019["Year"] = "2019"
        data2020["Year"] = "2020"
        data2021["Year"] = "2021"
        data2022["Year"] = "2022"
        data2023["Year"] = "2023"

        combined_data = pd.concat(
            [data2018, data2019, data2020, data2021, data2022, data2023]
        )
        combined_data.columns = ["Subject", "Count", "Year"]
        selected_subjects = ["MEDI", "ENGI", "PHYS", "BIOC", "COMP", "CHEM", "MATE"]
        filtered_data = combined_data[combined_data["Subject"].isin(selected_subjects)]
        stackData = filtered_data.pivot_table(
            index="Subject", columns="Year", values="Count", aggfunc="sum", fill_value=0
        )
        stackData = stackData.reset_index()
        years = st.multiselect(
            "Select Years",
            options=["2018", "2019", "2020", "2021", "2022", "2023"],
            default=["2018", "2019", "2020", "2021", "2022", "2023"],
        )

        if years:
            filtered_stack_data = stackData[["Subject"] + years]
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            data_transposed = filtered_stack_data.set_index("Subject").T
            data_transposed.plot(kind="barh", stacked=True, ax=ax1, colormap="Set2")

            ax1.set_title("Trend of Research Papers by Subject", fontsize=16)
            ax1.set_xlabel("Number of Papers", fontsize=12)
            ax1.set_ylabel("Year", fontsize=12)
            ax1.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(axis="x", linestyle="--", alpha=0.6)
            st.pyplot(fig1)
        else:
            st.warning("Please select at least one year to view the trends.")

        st.write("---")
        year_data_map = {
            "2018": data2018,
            "2019": data2019,
            "2020": data2020,
            "2021": data2021,
            "2022": data2022,
            "2023": data2023,
        }
        st.header("Trend of Top Subjects Over Years")
        subjectfull = st.selectbox(
            "Select Subject",
            [
                "Medical",
                "Engineer",
                "Physics",
                "Biochemistry",
                "Computer",
                "Chemistry",
                "Material",
            ],
        )
        subject_map = {
            "Medical": "MEDI",
            "Engineer": "ENGI",
            "Physics": "PHYS",
            "Biochemistry": "BIOC",
            "Computer": "COMP",
            "Chemistry": "CHEM",
            "Material": "MATE",
        }
        subject = subject_map.get(subjectfull)

        for year, data in year_data_map.items():
            data["subjectCode"] = data["subjectCode"].apply(
                lambda x: (
                    x
                    if isinstance(x, str)
                    else " ".join(x) if isinstance(x, list) else ""
                )
            )

        filtered_data = pd.concat(
            [
                year_data_map[year][
                    year_data_map[year]["subjectCode"].str.contains(
                        subject, case=False, na=False
                    )
                ].assign(Year=year)
                for year in years
            ]
        )

        data_summary = filtered_data.groupby("Year")["count"].sum().reset_index()

        st.write(f"### Description:")
        st.write(
            f"The chart below displays the trend of **{subjectfull}** papers over the selected years. "
            "The count of papers represents the number of times the selected subject appeared in each year's data."
        )
        fig = px.line(
            data_summary,
            x="Year",
            y="count",
            title=f"Trend of {subjectfull} Papers Over Years",
            line_shape="spline",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        st.plotly_chart(fig)
        st.write(f"### Statistical Summary {subjectfull} Papers:")
        min_count = data_summary["count"].min()
        max_count = data_summary["count"].max()
        mean_count = data_summary["count"].mean()
        std_dev_count = data_summary["count"].std()
        st.write(f"- **Minimum Count**: {min_count}")
        st.write(f"- **Maximum Count**: {max_count}")
        st.write(f"- **Mean Count**: {mean_count:.2f}")

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
                color="Subject Code",
                color_discrete_sequence=px.colors.qualitative.Set3,
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

            lang = pd.read_csv(f"data/languageData{year}.csv")
            # st.error("Selected year data is unavailable.")
            categories = lang["language.@xml:lang"].tolist()
            values = lang["count"].tolist()
            N = len(categories)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color="orange", alpha=0.25)
            ax.plot(angles, values, color="orange", linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            st.write("---")
            ax.set_title("Language Distribution of Paper (Except English)", fontsize=12)
            st.pyplot(fig)

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
        ["closeness", "degree", "betweenness", "pagerank"],
    )

    graph_size = st.sidebar.slider(
        "Graph Size",
        min_value=500,
        max_value=3000,
        value=1400,
        step=100,
    )

    # Adjust node spacing for spring layout
    if layout_option == "spring":
        node_spacing = st.sidebar.slider(
            "Node Spacing",
            min_value=1.0,
            max_value=20.0,
            value=6.0,
            step=1.0,
        )
    else:
        node_spacing = 2.0

    # Font size and style for node labels
    font_size = st.sidebar.slider(
        "Label Font Size",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
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
    elif centrality_option == "pagerank":
        centrality = nx.pagerank(G)
    else:
        centrality = nx.closeness_centrality(G)

    # Node sizes based on centrality measure
    node_sizes = [centrality[node] * 1000 for node in G.nodes]

    # Detect communities if requested
    st.markdown("---")  # Add separator
    show_communities = st.sidebar.checkbox("Detect Communities")
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

    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 8))  

    if communities:
        node_colors = [communities.get(node, -1) for node in G.nodes]
        unique_communities = list(set(node_colors))
        colormap = (
            plt.cm.tab10
        )
    else:
        node_colors = ["lightblue"] * len(G.nodes)
        # Gray colormap if no communities
        colormap = plt.cm.Greys 

    if show_edges:
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=colormap,
            font_size=font_size,
            font_family=font_style,
            font_weight="light",
            edge_color="lightgray",
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=colormap,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=font_size,
            font_weight="light",
        )

    st.pyplot(fig)

    st.subheader("Keyword Frequency Word Cloud")
    all_keywords = sum(sample_data["keyword"], [])
    word_freq = pd.Series(all_keywords).value_counts()
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    st.image(wordcloud.to_array(), use_container_width=True)

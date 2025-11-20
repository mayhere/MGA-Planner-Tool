import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV data
file_path = "C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/C3_2025-2035/Earth_to_Mercury_C3_RLA_DLA_2025_2035.csv"
df = pd.read_csv(file_path)

# Convert launch date
df["Launch Date"] = pd.to_datetime(df["Launch Date"], dayfirst=True)

# Filter to only include C3 ≤ 110
df = df[df["C3L (km²/s²)"] <= 100]

# Classify mission types based on C3 energy
df["Mission Type"] = df["C3L (km²/s²)"].apply(lambda x: "Flyby" if x < 85 else "Orbiter")

# Create base scatter plot
fig = px.scatter(
    df,
    x="Launch Date",
    y="C3L (km²/s²)",
    color="Mission Type",
    symbol="Mission Type",
    title="Earth-Mercury Launch Opportunities 2025-2035 C3L Plot",
    labels={"C3L (km²/s²)": "C3 Launch Energy (km²/s²)"},
    hover_data={
        "Launch Date": True,
        "C3L (km²/s²)": True,
        "Time of Flight (days)": True,
        "Mission Type": True
    },
    template="plotly_white"
)

# Highlight red stars for C3 between 40–90 for years 2026–27 and 2033–34
highlight_df = df[
    ((df["Launch Date"].dt.year.between(2026, 2027)) |
     (df["Launch Date"].dt.year.between(2033, 2034))) &
    (df["C3L (km²/s²)"].between(40, 90))
]

fig.add_trace(
    go.Scatter(
        x=highlight_df["Launch Date"],
        y=highlight_df["C3L (km²/s²)"],
        mode="markers",
        marker=dict(symbol="star", size=10, color="red"),
        name="Highlighted C3 (40–90, 2026–27 & 2033–34)"
    )
)

# Update layout
fig.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(
    xaxis_title="Launch Date",
    yaxis_title="C3L (km²/s²)",
    legend_title="Mission Type",
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

# Show and save
fig.show()
fig.write_html("C:/Users/Mayank/Desktop/VSSC-ISRO/Codes/C3_2025-2035/Earth_Mercury_C3L_2025-2035.html")

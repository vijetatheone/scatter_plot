import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Output, Input, State
import dash_bootstrap_components as dbc

# ---- Load Data ----
SCORES_FILE = "Percentile_check.xlsx"        # Industry, Section, Attribute, Brand, Score, Reason
RUBRICS_FILE = "Score Conditions.xlsx"       # Section, Attribute, Scoring Rubric (0–5)

scores_df = pd.read_excel(SCORES_FILE)
rubrics_df = pd.read_excel(RUBRICS_FILE)

# Ensure Reason is optional
if "Reason" not in scores_df.columns:
    scores_df["Reason"] = ""

# ✅ Include Industry now
scores_df = scores_df[['Industry','Section','Attribute','Brand','Score','Reason']].dropna()
rubrics_df = rubrics_df[['Section','Attribute','Scoring Rubric (0–5)']].dropna()

industries = sorted(scores_df['Industry'].unique())
sections = sorted(scores_df['Section'].unique())

# Merge rubrics into scores
merged_df = scores_df.merge(rubrics_df, on=['Section', 'Attribute'], how='left')

# ---- Dash App ----
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.title = "UX Dashboard with Industry Filter"

# ---- Chart Functions ----
def make_overall_chart(industry, section):
    sdf = merged_df[(merged_df['Industry']==industry) & (merged_df['Section']==section)]
    overall = sdf.groupby("Brand", as_index=False)['Score'].mean()

    # Shift scores so bubbles sit inside boxes
    overall['ScorePlot'] = overall['Score'] + 0.5 + np.random.uniform(-0.05, 0.05, len(overall))

    fig = px.scatter(
        overall,
        x="ScorePlot", y=[0]*len(overall),
        hover_name="Brand",
        hover_data={"Score":":.2f"},
        color="Score",
        color_continuous_scale="Viridis",
        range_x=[0, 5.5],
        title=f"{industry} — {section} Overall Brand Scores"
    )

    # ✅ fix marker size
    fig.update_traces(marker=dict(size=14))

    # Background bands
    band_colors = ["#ffcccc", "#ffe0b3", "#ffffb3", "#ccffcc", "#cce5ff"]
    for i in range(6):
        fig.add_vrect(
            x0=i, x1=i+1,
            fillcolor=band_colors[i % len(band_colors)],
            opacity=0.3,
            layer="below",
            line_width=0
        )

    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_xaxes(
        title="Score",
        tickmode="array",
        tickvals=[0.5, 1.5, 2.5, 3.5, 4.5],
        ticktext=["0","1","2","3","4","5"]
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10,r=10,t=50,b=40))
    return fig


def make_attribute_chart(industry, section):
    sdf = merged_df[(merged_df['Industry']==industry) & (merged_df['Section']==section)].copy()

    # Shift scores into boxes + jitter
    sdf['ScorePlot'] = sdf['Score'] + 0.5 + np.random.uniform(-0.1, 0.1, len(sdf))

    # Order attributes by mean score
    order = sdf.groupby('Attribute')['Score'].mean().sort_values().index.tolist()

    fig = px.scatter(
        sdf,
        x="ScorePlot", y="Attribute",
        hover_name="Brand",
        hover_data={"Score":":.2f","Reason":True},
        color="Score",
        color_continuous_scale="Viridis",
        range_x=[0, 5.5],
        title=f"{industry} — {section} Attribute Scores",
        height=max(500, 50*len(order))
    )

    # ✅ fix marker size
    fig.update_traces(marker=dict(size=12))

    # Background bands
    band_colors = ["#ffcccc", "#ffe0b3", "#ffffb3", "#ccffcc", "#cce5ff"]
    for i in range(6):
        fig.add_vrect(
            x0=i, x1=i+1,
            fillcolor=band_colors[i % len(band_colors)],
            opacity=0.3,
            layer="below",
            line_width=0
        )

    fig.update_layout(
        yaxis=dict(title=None, categoryorder='array', categoryarray=order),
        xaxis=dict(
            title="Score",
            tickmode="array",
            tickvals=[0.5, 1.5, 2.5, 3.5, 4.5],
            ticktext=["0","1","2","3","4","5"]
        ),
        coloraxis_showscale=False,
        margin=dict(l=20,r=20,t=50,b=40)
    )
    return fig


def make_rubric_cards():
    cards = []
    for _, row in rubrics_df.iterrows():
        card = dbc.Card(
            dbc.CardBody([
                html.H5(row['Attribute'], className="card-title"),
                html.H6(f"Section: {row['Section']}", className="card-subtitle text-muted"),
                html.P(row['Scoring Rubric (0–5)'], className="card-text")
            ]),
            className="mb-3 shadow-sm"
        )
        cards.append(card)
    return dbc.Row([dbc.Col(cards, width=12)])

# ---- Layout ----
app.layout = dbc.Container([
    html.H2("Benchmark UX Dashboard", className="text-center mt-3"),

    html.Div([
        html.Label("Select Industry:"),
        dcc.Dropdown(
            id="industry-dd",
            options=[{"label": i, "value": i} for i in industries],
            value=industries[0],
            clearable=False,
            style={"width":"300px"}
        ),
        html.Label("Select Section:", className="mt-2"),
        dcc.Dropdown(
            id="section-dd",
            options=[{"label": s, "value": s} for s in sections],
            value=sections[0],
            clearable=False,
            style={"width":"300px"}
        ),
    ], className="my-3"),

    html.H4("Overall Performance"),
    dcc.Graph(id="overall-graph"),

    html.H4("Attribute Performance"),
    dcc.Graph(id="attr-graph"),

    # CTA button
    dbc.Button("View Scoring Details", id="details-btn", color="primary", className="mt-3"),

    # Collapsible section
    dbc.Collapse(
        dbc.Card(dbc.CardBody([
            html.H4("Scoring Rubric Reference"),
            make_rubric_cards()
        ])),
        id="details-collapse",
        is_open=False
    )
], fluid=True)

# ---- Callbacks ----
@app.callback(
    Output("overall-graph","figure"),
    Output("attr-graph","figure"),
    Input("industry-dd","value"),
    Input("section-dd","value")
)
def update_charts(industry, section):
    return make_overall_chart(industry, section), make_attribute_chart(industry, section)


@app.callback(
    Output("details-collapse", "is_open"),
    Input("details-btn", "n_clicks"),
    State("details-collapse", "is_open"),
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# ---- Run ----
if __name__ == "__main__":
    app.run(debug=True)

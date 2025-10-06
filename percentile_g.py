import os
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Output, Input, State, ctx, ALL
import dash_bootstrap_components as dbc
from flask import request, jsonify
from openai import OpenAI
import json

# ---- Load Data ----
# ---- Load Data ----
SCORES_FILE = "Percentile_check.xlsx"        # Industry, Section, Attribute, Brand, Score, Reason
RUBRICS_FILE = "Score Conditions.xlsx"       # Section, Attribute, Scoring Rubric (0–5), Details

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
scores_df = pd.read_excel(SCORES_FILE)
rubrics_df = pd.read_excel(RUBRICS_FILE)

# Ensure Reason column exists
if "Reason" not in scores_df.columns:
    scores_df["Reason"] = ""

# ✅ Include Industry + merge with rubric details
scores_df = scores_df[['Industry','Section','Attribute','Brand','Score','Reason']].dropna()
rubrics_df = rubrics_df[['Section','Attribute','Scoring Rubric (0–5)','Details']].dropna()

merged_df = scores_df.merge(rubrics_df, on=['Section','Attribute'], how='left')

industries = sorted(scores_df['Industry'].unique())
sections = sorted(scores_df['Section'].unique())

# Merge rubrics
#merged_df = scores_df.merge(rubrics_df, on=['Section', 'Attribute'], how='left')

# ---- Dash App ----
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "UX Dashboard with Industry Filter"

# ---- Chart Functions ----
def make_overall_chart(industry, section):
    sdf = merged_df[(merged_df['Industry']==industry) & (merged_df['Section']==section)]
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    overall = sdf.groupby("Brand", as_index=False)['Score'].mean()

    # jitter slightly but clip to 0–5
    overall['ScorePlot'] = np.clip(
        overall['Score'] + np.random.uniform(-0.1, 0.1, len(overall)),
        0, 5
    )

    fig = px.scatter(
        overall,
        x="ScorePlot", y=[0]*len(overall),
        hover_name="Brand",
        hover_data={"Score":":.2f"},
        color="Score",
        color_continuous_scale="RdYlGn",
        range_color=[0,5],
        range_x=[0,5],
        title=f"{industry} — {section} Overall Brand Scores"
    )

    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))
    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_xaxes(
        title="Score",
        tickmode="array",
        tickvals=[0,1,2,3,4,5],
        ticktext=["0","1","2","3","4","5"]
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=20,r=20,t=50,b=40)
    )
    return fig


def make_attribute_chart(industry, section):
    sdf = merged_df[(merged_df['Industry']==industry) & (merged_df['Section']==section)].copy()
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    # ✅ Use Details column for one-liner
    def format_label(attr, detail):
        if pd.notna(detail) and str(detail).strip():
            return f"{attr}<br><span style='font-size:12px; color:gray;'>{detail}</span>"
        else:
            return attr

    sdf['AttrLabel'] = sdf.apply(lambda x: format_label(x['Attribute'], x['Details']), axis=1)

    # ✅ Jitter and clip inside 0–5
    sdf['ScorePlot'] = np.clip(
        sdf['Score'] + np.random.uniform(-0.2, 0.2, len(sdf)),
        0, 5
    )

    # ✅ Keep Excel order of attributes
    order = rubrics_df[rubrics_df['Section']==section]['Attribute'].tolist()
    sdf['AttrLabel'] = pd.Categorical(sdf['AttrLabel'], 
                                      categories=[format_label(a, d) for a,d in 
                                                  rubrics_df[rubrics_df['Section']==section][['Attribute','Details']].values],
                                      ordered=True)

    fig = px.scatter(
        sdf,
        x="ScorePlot", y="AttrLabel",
        hover_name="Brand",
        hover_data={"Score":":.2f","Reason":True},
        color="Score",
        color_continuous_scale="RdYlGn",
        range_color=[0,5],
        title=f"{industry} — {section} Attribute Scores",
        height=max(500, 90*len(order))  # more height per attribute
    )

    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))

    # ✅ Horizontal row-bands
    for i, attr in enumerate(order):
        fig.add_hrect(
            y0=i-0.5, y1=i+0.5,
            fillcolor="rgba(240,240,240,0.3)" if i % 2 == 0 else "rgba(255,255,255,0)",
            layer="below", line_width=0
        )

    fig.update_layout(
        yaxis=dict(title=None, categoryorder="array", categoryarray=[format_label(a, d) for a,d in rubrics_df[rubrics_df['Section']==section][['Attribute','Details']].values]),
        xaxis=dict(
            title="Score",
            range=[-0.2, 5.3],
            tickmode="array",
            tickvals=[0,1,2,3,4,5],
            ticktext=["0","1","2","3","4","5"]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=80,r=80,t=50,b=40)
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
    html.H2("Benchmark UX Dashboard", className="mt-3 text-center"),
    dcc.Store(id="screen-width"),
    dcc.Interval(id="resize-listener", interval=2000, n_intervals=0),

    # Industry + Section in one row
    dbc.Row([
        dbc.Col([
            html.Label("Select Industry:"),
            dcc.Dropdown(
                id="industry-dd",
                options=[{"label": i, "value": i} for i in industries],
                value=industries[0],
                clearable=False
            )
        ], width=6),
        dbc.Col([
            html.Label("Select Section:"),
            dbc.ButtonGroup(
                [dbc.Button(s,
                            id={"type":"section-btn","index":s},
                            outline=True, color="primary",
                            className="mx-1 rounded-pill") for s in sections],
                className="mb-2"
            )
        ], width=6)
    ], className="my-2"),

    # Right-aligned scoring details button
    html.Div([
        dbc.Button("View Scoring Details", id="details-btn", color="secondary")
    ], style={"textAlign":"right", "marginBottom":"15px"}),

    # Overall Performance card
    dbc.Card(
        dbc.CardBody([
            html.H4("Overall Performance"),
            dcc.Graph(id="overall-graph")
        ]),
        className="mb-4 shadow-sm p-3"
    ),

    # Attribute Performance card
    dbc.Card(
        dbc.CardBody([
            html.H4("Attribute Performance"),
            dcc.Graph(id="attr-graph")
        ]),
        className="mb-4 shadow-sm p-3"
    ),

    # Modal for scoring details
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Scoring Rubric Reference")),
            dbc.ModalBody(make_rubric_cards())
        ],
        id="details-modal",
        is_open=False,
        size="lg"
    )
], fluid=True)

# ---- Callbacks ----
@app.callback(
    Output("section-store","data"),
    Input({"type":"section-btn","index":ALL},"n_clicks"),
    prevent_initial_call=True
)
def update_section(n_clicks):
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered["index"]
    return sections[0]

# hidden store for section
app.layout.children.insert(2, dcc.Store(id="section-store", data=sections[0]))

@app.callback(
    Output("overall-graph","figure"),
    Output("attr-graph","figure"),
    Input("industry-dd","value"),
    Input("section-store","data")
)
def update_charts(industry, section):
    return make_overall_chart(industry, section), make_attribute_chart(industry, section)

@app.callback(
    Output("details-modal", "is_open"),
    Input("details-btn", "n_clicks"),
    State("details-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open
app.clientside_callback(
    """
    function(n) {
        return window.innerWidth;
    }
    """,
    Output("screen-width", "data"),
    Input("resize-listener", "n_intervals")
)

@server.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    brand = data.get("brand")
    industry = data.get("industry")
    section = data.get("section")
    image_url = data.get("image_url")

    if not all([brand, industry, section, image_url]):
        return jsonify({"error": "Missing one or more required fields"}), 400

    prompt = f"""
    You are a UX design evaluator for the {industry} industry.
    Analyze the uploaded screenshot of a {section} page and rate the following attributes 0–5,
    using the official rubric provided earlier. Return JSON strictly in this format:
    [
      {{"Attribute": "Hero Banner", "Score": 4, "Reason": "Visually appealing hero with CTA"}},
      ...
    ]
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a UX benchmarking expert."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )
        content = response.choices[0].message.content
        gpt_scores = json.loads(content)

        # Convert GPT output → DataFrame and append to dataset
        new_data = pd.DataFrame([
            {
                "Industry": industry,
                "Section": section,
                "Brand": brand,
                "Attribute": item["Attribute"],
                "Score": float(item["Score"]),
                "Reason": item.get("Reason", "")
            }
            for item in gpt_scores
        ])

        global merged_df
        merged_df = pd.concat([merged_df, new_data], ignore_index=True)

        return jsonify({"status": "ok", "brand": brand, "industry": industry, "section": section})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Run ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

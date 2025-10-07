import os
import json
import time
import threading
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Output, Input, State, ctx, ALL
import dash_bootstrap_components as dbc
from flask import request, jsonify, send_file
from openai import OpenAI

# ---- Setup ----
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TEMP_DIR = "temp_reports"
MASTER_FILE = "scores_master.json"
os.makedirs(TEMP_DIR, exist_ok=True)

SCORES_FILE = "Percentile_check.xlsx"
RUBRICS_FILE = "Score Conditions.xlsx"

# ---- Load Base Data ----
rubrics_df = pd.read_excel(RUBRICS_FILE)[['Section','Attribute','Scoring Rubric (0–5)','Details']].dropna()

if os.path.exists(MASTER_FILE):
    scores_df = pd.read_json(MASTER_FILE)
else:
    scores_df = pd.read_excel(SCORES_FILE)[['Industry','Section','Attribute','Brand','Score','Reason']].dropna()
    scores_df.to_json(MASTER_FILE, orient="records")

merged_df = scores_df.merge(rubrics_df, on=['Section','Attribute'], how='left')
industries = sorted(scores_df['Industry'].unique())
sections = sorted(scores_df['Section'].unique())

# ---- Dash App ----
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "UX Dashboard (Hybrid Model)"

# ---- Visualization Functions ----
def make_overall_chart(industry, section, highlight_brand=None, df=None):
    # use the provided df (viz data) if present, else fall back to global
    use_df = df if df is not None else merged_df

    sdf = use_df[(use_df['Industry'] == industry) & (use_df['Section'] == section)]
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    overall = sdf.groupby("Brand", as_index=False)['Score'].mean()

    # ensure numeric; avoid discrete legend behavior
    overall['Score'] = pd.to_numeric(overall['Score'], errors='coerce')
    overall = overall.dropna(subset=['Score'])

    # jitter for readability, but keep inside 0–5
    overall['ScorePlot'] = np.clip(overall['Score'] + np.random.uniform(-0.1, 0.1, len(overall)), 0, 5)

    fig = px.scatter(
        overall,
        x="ScorePlot",
        y=[0] * len(overall),
        hover_name="Brand",
        hover_data={"Score": ":.2f"},
        color="Score",                         # continuous color by score
        color_continuous_scale="RdYlGn",
        range_color=[0, 5],
        range_x=[0, 5],
        title=f"{industry} — {section} Overall Brand Scores"
    )

    # base styling
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))

    # highlight uploaded brand with a star + label
    if highlight_brand and (highlight_brand in overall['Brand'].values):
        h = overall[overall['Brand'] == highlight_brand].iloc[0]
        fig.add_scatter(
            x=[h['ScorePlot']], y=[0],
            mode='markers+text',
            marker=dict(size=22, color='black', symbol='star', line=dict(width=2, color='black')),
            text=[highlight_brand],
            textposition="top center",
            name="Uploaded Brand",
            showlegend=False
        )

    # axes & layout
    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_xaxes(title="Score", tickmode="array", tickvals=[0,1,2,3,4,5], ticktext=["0","1","2","3","4","5"])
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,   # hide color scale
        showlegend=False,            # hide legend
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig


def make_attribute_chart(industry, section, highlight_brand=None, df=None):
    use_df = df if df is not None else merged_df

    sdf = use_df[(use_df['Industry'] == industry) & (use_df['Section'] == section)].copy()
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    # ensure numeric
    sdf['Score'] = pd.to_numeric(sdf['Score'], errors='coerce')
    sdf = sdf.dropna(subset=['Score'])

    # merge details (if not already there)
    if 'Details' not in sdf.columns:
        sdf = sdf.merge(rubrics_df[['Section','Attribute','Details']], on=['Section','Attribute'], how='left')

    def label(attr, detail):
        if pd.notna(detail) and str(detail).strip():
            return f"{attr}<br><span style='font-size:12px; color:gray;'>{detail}</span>"
        return attr

    # label + jitter
    sdf['AttrLabel'] = sdf.apply(lambda x: label(x['Attribute'], x.get('Details', None)), axis=1)
    sdf['ScorePlot'] = np.clip(sdf['Score'] + np.random.uniform(-0.2, 0.2, len(sdf)), 0, 5)

    # keep rubric order
    order = rubrics_df[rubrics_df['Section'] == section]['Attribute'].tolist()
    sdf['AttrLabel'] = pd.Categorical(
        sdf['AttrLabel'],
        categories=[label(a, d) for a, d in rubrics_df[rubrics_df['Section'] == section][['Attribute', 'Details']].values],
        ordered=True
    )

    fig = px.scatter(
        sdf,
        x="ScorePlot",
        y="AttrLabel",
        hover_name="Brand",
        hover_data={"Score": ":.2f", "Reason": True},
        color="Score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 5],
        title=f"{industry} — {section} Attribute Scores",
        height=max(500, 90 * len(order))
    )

    # base markers
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))

    # row bands
    for i, _ in enumerate(order):
        fig.add_hrect(
            y0=i-0.5, y1=i+0.5,
            fillcolor="rgba(240,240,240,0.3)" if i % 2 == 0 else "rgba(255,255,255,0)",
            layer="below", line_width=0
        )

    # highlight uploaded brand
    if highlight_brand and (highlight_brand in sdf['Brand'].values):
        hf = sdf[sdf['Brand'] == highlight_brand]
        fig.add_scatter(
            x=hf['ScorePlot'],
            y=hf['AttrLabel'],
            mode='markers+text',
            marker=dict(size=20, color='black', symbol='star', line=dict(width=2, color='black')),
            text=[highlight_brand] * len(hf),
            textposition="top center",
            name="Uploaded Brand",
            showlegend=False
        )

    fig.update_layout(
        yaxis=dict(title=None, categoryorder="array",
                   categoryarray=[label(a, d) for a, d in rubrics_df[rubrics_df['Section'] == section][['Attribute','Details']].values]),
        xaxis=dict(title="Score", range=[-0.2, 5.3],
                   tickmode="array", tickvals=[0,1,2,3,4,5], ticktext=["0","1","2","3","4","5"]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,   # hide color bar
        showlegend=False,            # hide legend
        margin=dict(l=80, r=80, t=50, b=40)
    )
    return fig


# ---- Layout ----
app.layout = dbc.Container([
    html.H2("Benchmark UX Dashboard", className="mt-3 text-center"),
    dcc.Store(id="section-store", data=sections[0]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Industry:"),
            dcc.Dropdown(id="industry-dd", options=[{"label":i,"value":i} for i in industries], value=industries[0], clearable=False)
        ], width=6),
        dbc.Col([
            html.Label("Select Section:"),
            dbc.ButtonGroup([dbc.Button(s, id={"type":"section-btn","index":s}, outline=True, color="primary", className="mx-1 rounded-pill") for s in sections])
        ], width=6)
    ], className="my-2"),

    dbc.Card(dbc.CardBody([
        html.H4("Overall Performance"),
        dcc.Graph(id="overall-graph")
    ]), className="mb-4 shadow-sm"),

    dbc.Card(dbc.CardBody([
        html.H4("Attribute Performance"),
        dcc.Graph(id="attr-graph")
    ]), className="mb-4 shadow-sm")
], fluid=True)

# ---- Callbacks ----
@app.callback(
    Output("section-store","data"),
    Input({"type":"section-btn","index":ALL},"n_clicks"),
    prevent_initial_call=True
)
def update_section(n_clicks):
    trig = ctx.triggered_id
    if trig and isinstance(trig, dict):
        return trig["index"]
    return sections[0]

@app.callback(
    Output("overall-graph","figure"),
    Output("attr-graph","figure"),
    Input("industry-dd","value"),
    Input("section-store","data")
)
def update_charts(industry, section):
    return make_overall_chart(industry, section), make_attribute_chart(industry, section)

# ---- GPT Analysis (Async Hybrid) ----
def run_analysis(data, temp_path):
    try:
        brand, industry, section, image_url = data["brand"], data["industry"], data["section"], data["image_url"]
        section_attrs = rubrics_df[rubrics_df["Section"].str.lower()==section.lower()]

        attr_text = "\n".join([
            f"Attribute: {r['Attribute']}\nScoring Rubric: {r['Scoring Rubric (0–5)']}\nDetails: {r['Details']}"
            for _,r in section_attrs.iterrows()
        ])

        prompt = f"""
        Evaluate this {industry} website screenshot for {section}.
        Score each attribute (0–5) based on rubric below, output JSON array with fields: Industry, Section, Attribute, Brand, Score, Reason.
        {attr_text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a strict UX benchmarking evaluator."},
                {"role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":image_url}}
                ]}
            ],
            timeout=120
        )

        output = response.choices[0].message.content.strip()
        try:
            result_json = eval(output)
        except Exception:
            result_json = [{"Attribute":"ParseError","Reason":output,"Score":None}]

        # Save temporary file
        with open(temp_path,"w") as f:
            json.dump({"status":"done","scores":result_json,"brand":brand,"industry":industry,"section":section},f)

        # Append to master
        df = pd.DataFrame(result_json)
        df["Industry"], df["Section"], df["Brand"] = industry, section, brand
        global merged_df, scores_df
        scores_df = pd.concat([scores_df, df], ignore_index=True)
        scores_df.to_json(MASTER_FILE, orient="records")
        merged_df = scores_df.merge(rubrics_df, on=["Section","Attribute"], how="left")

    except Exception as e:
        with open(temp_path,"w") as f:
            json.dump({"status":"error","error":str(e)},f)

@server.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    job_id = f"temp_{int(time.time()*1000)}"
    temp_path = os.path.join(TEMP_DIR, f"{job_id}.json")

    threading.Thread(target=run_analysis, args=(data, temp_path), daemon=True).start()
    return jsonify({"job_id":job_id, "status":"processing"})

@server.route("/report_status", methods=["GET"])
def report_status():
    job_id = request.args.get("job_id")
    temp_path = os.path.join(TEMP_DIR, f"{job_id}.json")
    if not os.path.exists(temp_path):
        return jsonify({"status":"processing"})
    with open(temp_path) as f:
        return jsonify(json.load(f))

@server.route("/temp_report", methods=["GET"])
def temp_report():
    job_id = request.args.get("job_id")
    temp_path = os.path.join(TEMP_DIR, f"{job_id}.json")

    if not os.path.exists(temp_path):
        return jsonify({"error": "Report not ready"}), 404

    with open(temp_path) as f:
        data = json.load(f)

    brand = data.get("brand")
    industry = data.get("industry")
    section = data.get("section")
    scores = data.get("scores", [])

    # temp DF from GPT result
    temp_df = pd.DataFrame(scores)
    temp_df["Industry"] = industry
    temp_df["Section"] = section
    temp_df["Brand"] = brand

    # combine with master for visualization
    viz_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # generate BOTH charts using viz_df, and highlight uploaded brand
    overall_fig = make_overall_chart(industry, section, highlight_brand=brand, df=viz_df)
    attr_fig    = make_attribute_chart(industry, section, highlight_brand=brand, df=viz_df)

    # simple page with both charts
    html_page = f"""
    <html>
      <head>
        <title>{industry} – {section} (Brand: {brand})</title>
        <meta charset="UTF-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
          body {{
            font-family: Inter, system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
            margin: 24px;
            color: #0a2239;
            background: #fff;
          }}
          h2 {{ margin: 0 0 8px 0; }}
          h3 {{ margin: 24px 0 8px 0; }}
        </style>
      </head>
      <body>
        <h2>{industry} — {section} (Brand: {brand})</h2>

        <h3>Overall Performance</h3>
        {overall_fig.to_html(full_html=False, include_plotlyjs=False)}

        <h3>Attribute-wise Performance</h3>
        {attr_fig.to_html(full_html=False, include_plotlyjs=False)}
      </body>
    </html>
    """
    return html_page

# ---- Run ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT",8050))
    app.run(host="0.0.0.0", port=port, debug=False)

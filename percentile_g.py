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
def make_overall_chart(industry, section, highlight_brand=None):
    sdf = merged_df[(merged_df['Industry']==industry) & (merged_df['Section']==section)]
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    overall = sdf.groupby("Brand", as_index=False)['Score'].mean()
    overall['ScorePlot'] = np.clip(overall['Score'] + np.random.uniform(-0.1,0.1,len(overall)),0,5)

    fig = px.scatter(
        overall, x="ScorePlot", y=[0]*len(overall),
        hover_name="Brand", hover_data={"Score":":.2f"},
        color="Score", color_continuous_scale="RdYlGn", range_color=[0,5],
        range_x=[0,5], title=f"{industry} — {section} Overall Brand Scores"
    )

    # Highlight the current uploaded brand
    if highlight_brand and highlight_brand in overall['Brand'].values:
        highlight = overall[overall['Brand']==highlight_brand]
        fig.add_scatter(
            x=highlight['ScorePlot'], y=[0], mode="markers+text",
            marker=dict(size=20, color="black", symbol="star"),
            text=highlight_brand, textposition="top center", name="Uploaded Brand"
        )

    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(title="Score", tickvals=[0,1,2,3,4,5])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20,r=20,t=50,b=40))
    return fig


def make_attribute_chart(industry, section, highlight_brand=None):
    sdf = merged_df[(merged_df['Industry']==industry) & (merged_df['Section']==section)].copy()
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    def label(attr, detail):
        return f"{attr}<br><span style='font-size:12px; color:gray;'>{detail}</span>" if pd.notna(detail) else attr

    sdf['AttrLabel'] = sdf.apply(lambda x: label(x['Attribute'], x['Details']), axis=1)
    sdf['ScorePlot'] = np.clip(sdf['Score'] + np.random.uniform(-0.2,0.2,len(sdf)), 0, 5)

    order = rubrics_df[rubrics_df['Section']==section]['Attribute'].tolist()
    sdf['AttrLabel'] = pd.Categorical(sdf['AttrLabel'], 
                                      categories=[label(a,d) for a,d in rubrics_df[rubrics_df['Section']==section][['Attribute','Details']].values],
                                      ordered=True)

    fig = px.scatter(
        sdf, x="ScorePlot", y="AttrLabel", hover_name="Brand",
        hover_data={"Score":":.2f","Reason":True},
        color="Score", color_continuous_scale="RdYlGn", range_color=[0,5],
        title=f"{industry} — {section} Attribute Scores",
        height=max(500,90*len(order))
    )

    # Highlight new brand
    if highlight_brand:
        hf = sdf[sdf['Brand']==highlight_brand]
        fig.add_scatter(
            x=hf['ScorePlot'], y=hf['AttrLabel'],
            mode="markers", marker=dict(size=18, color="black", symbol="star"),
            name="Uploaded Brand"
        )

    for i, _ in enumerate(order):
        fig.add_hrect(y0=i-0.5, y1=i+0.5, fillcolor="rgba(240,240,240,0.3)" if i%2==0 else "rgba(255,255,255,0)", layer="below", line_width=0)

    fig.update_layout(
        yaxis=dict(title=None, categoryorder="array", categoryarray=[label(a,d) for a,d in rubrics_df[rubrics_df['Section']==section][['Attribute','Details']].values]),
        xaxis=dict(title="Score", range=[-0.2,5.3], tickvals=[0,1,2,3,4,5]),
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=80,r=80,t=50,b=40)
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

    brand, industry, section = data.get("brand"), data.get("industry"), data.get("section")
    scores = data.get("scores", [])

    # --- Create temporary DataFrame for uploaded brand ---
    temp_df = pd.DataFrame(scores)
    temp_df["Industry"] = industry
    temp_df["Section"] = section
    temp_df["Brand"] = brand

    # --- Append temporarily to merged_df for visualization ---
    viz_df = pd.concat([merged_df, temp_df], ignore_index=True)
    viz_df = viz_df.merge(rubrics_df, on=["Section", "Attribute"], how="left")

    # --- Generate both charts ---
    overall_fig = make_overall_chart(industry, section, highlight_brand=brand)
    attr_fig = make_attribute_chart(industry, section, highlight_brand=brand)

    # --- Combine both charts into one HTML page ---
    html_page = f"""
    <html>
      <head>
        <title>{industry} – {section} Report</title>
        <meta charset="UTF-8">
        <style>
          body {{
            font-family: Inter, sans-serif;
            margin: 20px;
            background: #fff;
            color: #0a2239;
          }}
          h2 {{
            margin-bottom: 10px;
          }}
          iframe {{
            width: 100%;
            height: 600px;
            border: none;
            margin-bottom: 40px;
          }}
        </style>
      </head>
      <body>
        <h2>{industry} — {section} (Brand: {brand})</h2>
        <h3>Overall Performance</h3>
        {overall_fig.to_html(full_html=False, include_plotlyjs='cdn')}
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

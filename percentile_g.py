import os
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Output, Input, State, ctx, ALL
import dash_bootstrap_components as dbc
from flask import request, jsonify
from openai import OpenAI
import json
import urllib.parse
from flask import request, jsonify, render_template_string
import uuid
from datetime import datetime, timedelta

# === CONFIGURATION ===
SCORES_FILE = "Percentile_check.xlsx"        # Industry, Section, Attribute, Brand, Score, Reason
RUBRICS_FILE = "Score Conditions.xlsx"       # Section, Attribute, Scoring Rubric (0–5), Details
# global in-memory temporary reports
temp_reports = {}
TEMP_EXPIRY_MINUTES = 15
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === LOAD DATA ===
scores_df = pd.read_excel(SCORES_FILE)
rubrics_df = pd.read_excel(RUBRICS_FILE)

if "Reason" not in scores_df.columns:
    scores_df["Reason"] = ""

scores_df = scores_df[['Industry', 'Section', 'Attribute', 'Brand', 'Score', 'Reason']].dropna()
rubrics_df = rubrics_df[['Section', 'Attribute', 'Scoring Rubric (0–5)', 'Details']].dropna()

merged_df = scores_df.merge(rubrics_df, on=['Section', 'Attribute'], how='left')

industries = sorted(scores_df['Industry'].unique())
sections = sorted(scores_df['Section'].unique())

# === DASH APP ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "UX Dashboard with Industry Filter"


# === HELPER: RELOAD DATA ===
def reload_data():
    global scores_df, merged_df, industries, sections
    scores_df = pd.read_excel(SCORES_FILE)
    if "Reason" not in scores_df.columns:
        scores_df["Reason"] = ""
    scores_df = scores_df[['Industry', 'Section', 'Attribute', 'Brand', 'Score', 'Reason']].dropna()
    merged_df = scores_df.merge(rubrics_df, on=['Section', 'Attribute'], how='left')
    industries = sorted(scores_df['Industry'].unique())
    sections = sorted(scores_df['Section'].unique())


# === CHART FUNCTIONS ===
def make_overall_chart(industry, section, highlight_brand=None):
    sdf = merged_df[(merged_df['Industry'] == industry) & (merged_df['Section'] == section)]
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    overall = sdf.groupby("Brand", as_index=False)['Score'].mean()
    overall['ScorePlot'] = np.clip(overall['Score'] + np.random.uniform(-0.1, 0.1, len(overall)), 0, 5)

    fig = px.scatter(
        overall,
        x="ScorePlot", y=[0] * len(overall),
        hover_name="Brand",
        hover_data={"Score": ":.2f"},
        color="Score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 5],
        range_x=[0, 5],
        title=f"{industry} — {section} Overall Brand Scores"
    )

    # Highlight uploaded brand
    if highlight_brand and highlight_brand in overall["Brand"].values:
        highlight = overall[overall["Brand"] == highlight_brand]
        fig.add_scatter(
            x=highlight["ScorePlot"],
            y=[0],
            mode="markers+text",
            text=highlight["Brand"],
            textposition="top center",
            marker=dict(size=22, color="red", line=dict(width=2, color="black")),
            name=f"Highlighted: {highlight_brand}"
        )

    fig.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))
    fig.update_yaxes(showticklabels=False, title=None)
    fig.update_xaxes(title="Score", tickmode="array", tickvals=[0, 1, 2, 3, 4, 5])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False,
                      margin=dict(l=20, r=20, t=50, b=40))
    return fig


def make_attribute_chart(industry, section, highlight_brand=None):
    sdf = merged_df[(merged_df['Industry'] == industry) & (merged_df['Section'] == section)].copy()
    if sdf.empty:
        return px.scatter(title=f"No data for {industry} — {section}")

    def format_label(attr, detail):
        if pd.notna(detail) and str(detail).strip():
            return f"{attr}<br><span style='font-size:12px; color:gray;'>{detail}</span>"
        else:
            return attr

    sdf['AttrLabel'] = sdf.apply(lambda x: format_label(x['Attribute'], x['Details']), axis=1)
    sdf['ScorePlot'] = np.clip(sdf['Score'] + np.random.uniform(-0.2, 0.2, len(sdf)), 0, 5)

    order = rubrics_df[rubrics_df['Section'] == section]['Attribute'].tolist()
    sdf['AttrLabel'] = pd.Categorical(
        sdf['AttrLabel'],
        categories=[format_label(a, d) for a, d in rubrics_df[rubrics_df['Section'] == section][['Attribute', 'Details']].values],
        ordered=True
    )

    fig = px.scatter(
        sdf,
        x="ScorePlot", y="AttrLabel",
        hover_name="Brand",
        hover_data={"Score": ":.2f", "Reason": True},
        color="Score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 5],
        title=f"{industry} — {section} Attribute Scores",
        height=max(500, 90 * len(order))
    )

    # Highlight uploaded brand
    if highlight_brand and highlight_brand in sdf["Brand"].values:
        highlight = sdf[sdf["Brand"] == highlight_brand]
        fig.add_scatter(
            x=highlight["ScorePlot"],
            y=highlight["AttrLabel"],
            mode="markers+text",
            text=highlight["Brand"],
            textposition="top center",
            marker=dict(size=22, color="red", line=dict(width=2, color="black")),
            name=f"Highlighted: {highlight_brand}"
        )

    for i, attr in enumerate(order):
        fig.add_hrect(
            y0=i - 0.5, y1=i + 0.5,
            fillcolor="rgba(240,240,240,0.3)" if i % 2 == 0 else "rgba(255,255,255,0)",
            layer="below", line_width=0
        )

    fig.update_layout(
        yaxis=dict(title=None, categoryorder="array",
                   categoryarray=[format_label(a, d) for a, d in
                                  rubrics_df[rubrics_df['Section'] == section][['Attribute', 'Details']].values]),
        xaxis=dict(title="Score", range=[-0.2, 5.3], tickmode="array", tickvals=[0, 1, 2, 3, 4, 5]),
        plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False,
        margin=dict(l=80, r=80, t=50, b=40)
    )

    return fig


# === LAYOUT ===
app.layout = dbc.Container([
    html.H2("Benchmark UX Dashboard", className="mt-3 text-center"),
    dcc.Store(id="screen-width"),
    dcc.Interval(id="resize-listener", interval=2000, n_intervals=0),
    dcc.Store(id="section-store", data=sections[0]),

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
                [dbc.Button(s, id={"type": "section-btn", "index": s},
                            outline=True, color="primary", className="mx-1 rounded-pill")
                 for s in sections],
                className="mb-2"
            )
        ], width=6)
    ], className="my-2"),

    html.Div([
        dbc.Button("View Scoring Details", id="details-btn", color="secondary")
    ], style={"textAlign": "right", "marginBottom": "15px"}),

    dbc.Card(dbc.CardBody([html.H4("Overall Performance"), dcc.Graph(id="overall-graph")]),
             className="mb-4 shadow-sm p-3"),

    dbc.Card(dbc.CardBody([html.H4("Attribute Performance"), dcc.Graph(id="attr-graph")]),
             className="mb-4 shadow-sm p-3"),

    dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Scoring Rubric Reference")),
               dbc.ModalBody(html.Div("Rubric details available in 'Score Conditions.xlsx'"))],
              id="details-modal", is_open=False, size="lg")
], fluid=True)


# === CALLBACKS ===
@app.callback(
    Output("section-store", "data"),
    Input({"type": "section-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def update_section(n_clicks):
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered["index"]
    return sections[0]


@app.callback(
    Output("overall-graph", "figure"),
    Output("attr-graph", "figure"),
    Input("industry-dd", "value"),
    Input("section-store", "data")
)
def update_charts(industry, section):
    query = request.environ.get("QUERY_STRING", "")
    params = urllib.parse.parse_qs(query)
    highlight = params.get("highlight", [None])[0]
    return make_overall_chart(industry, section, highlight), make_attribute_chart(industry, section, highlight)


@app.callback(
    Output("details-modal", "is_open"),
    Input("details-btn", "n_clicks"),
    State("details-modal", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open


# === /analyze ENDPOINT ===
@app.server.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        brand = data.get("brand")
        industry = data.get("industry")
        section = data.get("section")
        image_url = data.get("image_url")

        rubric_df = pd.read_excel(RUBRICS_FILE)
        rubric_df = rubric_df.dropna(subset=["Section", "Attribute", "Scoring Rubric (0–5)", "Details"])
        section_attrs = rubric_df[rubric_df["Section"].str.lower() == section.lower()]

        if section_attrs.empty:
            return jsonify({"error": f"No rubric found for section '{section}'"}), 400

        # Build rubric prompt text
        attr_text = "\n".join([
            f"Attribute: {row['Attribute']}\nScoring Rubric: {row['Scoring Rubric (0–5)']}\nDetails: {row['Details']}"
            for _, row in section_attrs.iterrows()
        ])

        prompt = f"""
        You are a UI/UX benchmarking evaluator for the {industry} industry.
        Evaluate the uploaded webpage screenshot for the section: {section}.
        Use ONLY the attributes and rubrics given below. 
        Each attribute must receive one score (0–5) and a short reason.
        Output strictly as JSON array in this exact structure:
        [
          {{'Industry': '{industry}', 'Section': '{section}', 'Attribute': '<attribute>', 'Brand': '{brand}', 'Score': <score>, 'Reason': '<reason>'}},
          ...
        ]

        {attr_text}
        """

        # Call GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a visual UX benchmarking evaluator."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )

        output = response.choices[0].message.content.strip()
        try:
            result_json = eval(output)
        except Exception:
            result_json = [{"Attribute": "ParseError", "Reason": output, "Score": None}]

        # Temporary DataFrame
        df_temp = pd.DataFrame(result_json)
        df_temp["Industry"] = industry
        df_temp["Section"] = section
        df_temp["Brand"] = brand

        # Store in-memory
        key = str(uuid.uuid4())
        temp_reports[key] = {
            "data": df_temp,
            "created_at": datetime.utcnow()
        }

        # Return link
        return jsonify({
            "status": "success",
            "brand": brand,
            "section": section,
            "industry": industry,
            "scores": result_json,
            "report_url": f"https://scatter-plot.onrender.com/temp_report?key={key}"
        })

    except Exception as e:
        print("❌ Error in /analyze:", e)
        return jsonify({"error": str(e)}), 500


@app.server.route("/temp_report")
def temp_report():
    key = request.args.get("key")
    if not key or key not in temp_reports:
        return "⚠️ Report expired or invalid key.", 404

    # Cleanup expired entries
    now = datetime.utcnow()
    expired = [k for k, v in temp_reports.items()
               if now - v["created_at"] > timedelta(minutes=TEMP_EXPIRY_MINUTES)]
    for k in expired:
        del temp_reports[k]

    df_temp = temp_reports[key]["data"]
    industry = df_temp["Industry"].iloc[0]
    section = df_temp["Section"].iloc[0]
    brand = df_temp["Brand"].iloc[0]

    # Merge with master dataset (no persistence)
    combined = pd.concat([merged_df, df_temp], ignore_index=True)

    # Pass this combined data to charts
    # Add highlight logic
    fig_overall = make_overall_chart(industry, section)
    fig_attr = make_attribute_chart(industry, section)

    # highlight logic - differentiate the temp brand bubble
    for fig in [fig_overall, fig_attr]:
        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='black'),
                opacity=0.8
            )
        )
        fig.for_each_trace(
            lambda trace: trace.update(marker_color='red')
            if trace.name == brand else None
        )

    # Render inline HTML
    html_content = f"""
    <html>
    <head>
        <title>{brand} Temporary UX Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style='font-family: Arial; margin:40px;'>
        <h2>{brand} — {industry} / {section} Temporary Report</h2>
        <div id='overall'></div>
        <div id='attr'></div>
        <p><i>This report is temporary and will auto-expire in {TEMP_EXPIRY_MINUTES} minutes.</i></p>
        <script>
            var overall = {fig_overall.to_json()};
            var attr = {fig_attr.to_json()};
            Plotly.newPlot('overall', overall.data, overall.layout);
            Plotly.newPlot('attr', attr.data, attr.layout);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)


# === RUN APP ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)

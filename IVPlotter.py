from dash import Dash, html, dcc, callback, Output, Input, State  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import pandas as pd
import base64
import io
import plotly.io as pio  # type: ignore
import names  # type: ignore

pd.options.plotting.backend = "plotly"

# Initialize the app
app = Dash(__name__)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Label("Enter Plot Name:"),
        dcc.Input(id="plot-name", type="text", value="My Plot"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        dcc.Checklist(
            id="file-checklist",
        ),
        html.Div(id="output-data-upload"),
        html.Button("Download as png", id="btn_png"),
        dcc.Download(id="download-plot-png"),
        dcc.Store(id="figure-store", storage_type="memory"),
    ]
)


class IVdata:
    def __init__(self, df1=pd.DataFrame(), df2=pd.DataFrame()):
        self.data = df1
        self.metadata = df2

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.metadata

    def add(self, IVdat):
        self.data = pd.concat([self.data, IVdat.data])
        self.metadata = pd.concat([self.metadata, IVdat.metadata])


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return "Unset"


def Hier(df):
    # Add a new level to the index using groupby and cumcount
    df.set_index([df.groupby(level=0).cumcount(), df.index], append=True, inplace=True)

    # Reset the index to keep only two levels
    df.reset_index(level=[2], drop=True, inplace=True)
    return


def Readivdat(rawdata, mname):
    data = []
    START = False
    FOOT = False
    id = names.get_full_name()
    metadata = []
    mlabel = []
    DUTID = ""
    data_label = ""
    for line in rawdata:
        if "END FOOTER" in line:
            break
        if "END DATA" in line:
            df_data = pd.DataFrame(data, columns=data_label)
            df_data["Name"] = mname  # Add Measurement Name
            START = False
        if START:
            line_data = line.strip().split("\t")
            data.append(list(map(lambda x: float(x), line_data)))
        if "START DATA" in line:
            START = True
            data_label = next(rawdata).strip().split("\t")
        if "DUTID" in line:
            DUTID = line.split(":")
            DUTID = DUTID[-1].strip().strip('""')
        if "Source(" in line:
            source_index = line.index("Source(") + len("Source(")
            output_index = line.index("Output(%):", source_index)
            end_index = line.index("/I(A):", output_index)
            input_power = float(line[output_index + len("Output(%):") : end_index])
            input_power_value = str(input_power) + "%"
        if FOOT:
            mdata = line.split(":")
            mlabel.append(mdata[0].strip().strip('""'))
            metadata.append(to_float(mdata[1].strip().strip('""')))
        if "START FOOTER" in line:
            FOOT = True
    if data_label == "":
        return None
    df_metadata = pd.DataFrame([metadata], columns=mlabel)
    df_metadata["InputPower"] = input_power_value  # Add InputPower column
    df_metadata["Name"] = mname  # Add Measurement Name
    df_metadata["id"] = id
    df_metadata["DUTID"] = DUTID
    df = pd.DataFrame(data, columns=data_label)
    df["Name"] = mname  # Add Measurement Name
    df["id"] = id
    DF = IVdata(df, df_metadata)
    return DF


def PlotIV(DF, title):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Get unique values of InputPower for color mapping
    input_power_values = DF.metadata["InputPower"].unique()

    # Get unique names of measurements
    ids = DF.metadata["id"].unique()

    # Define a color scale using Plotly Express
    color_scale = px.colors.qualitative.Set1

    # Create a dictionary to map 'measure' names to colors
    measure_colors = {}
    for idx, measure in enumerate(ids):
        color = color_scale[idx % len(color_scale)]
        measure_colors[measure] = color

    # Iterate through unique InputPower values
    for idx, input_power in enumerate(input_power_values):
        legend_group = f"input_power_{input_power}"
        for measure in DF.metadata[DF.metadata["InputPower"] == input_power][
            "id"
        ].unique():
            color = measure_colors[measure]
            dfp = DF.data.loc[measure]
            current_trace = go.Scatter(
                x=dfp["Voltage(V)"],
                y=dfp["Current(A)"],
                name="Current(A)",
                line=dict(color=color),
                legendgroup=legend_group,  # Assign legend group
                legendgrouptitle_text=f"Input Power {input_power}",
            )

            power_trace = go.Scatter(
                x=dfp["Voltage(V)"],
                y=dfp["Power(W)"],
                name="Power(W)",
                line=dict(color=color, dash="dot"),
                legendgroup=legend_group,  # Assign legend group
            )

            fig.add_trace(current_trace, secondary_y=False)
            fig.add_trace(power_trace, secondary_y=True)

    # Add figure title
    fig.update_layout(title_text=title)

    # Set x-axis title
    fig.update_xaxes(title_text="Voltage")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Current</b> (A)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Power</b> (W)", secondary_y=True)
    fig.update_xaxes(title_text="<b>Voltage</b> (V)")

    # Enable legend clickmode to hide/show traces in the same legend group
    fig.update_layout(
        legend=dict(itemclick="toggle"),
        title={
            "text": title,
            #'y':0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=50),
        },
    )
    return fig


def Process_Files(list_of_contents, list_of_names, plotname):
    d = IVdata()
    for content, mname in zip(list_of_contents, list_of_names):
        _, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        rawdata = io.StringIO(decoded.decode("utf-8"))
        data = Readivdat(rawdata, mname)
        data.data["Current(A)"] *= -1
        data.data["Power(W)"] *= -1
        d.add(data)
    d.data.set_index("id", inplace=True)
    Hier(d.data)
    fig = PlotIV(d, plotname)
    return html.Div([dcc.Graph(figure=fig)]), fig


@callback(
    Output("output-data-upload", "children"),
    Output("figure-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input("plot-name", "value"),
)
def update_output(list_of_contents, list_of_names, plotname):
    fig = None
    if list_of_contents is not None:
        children, fig = Process_Files(list_of_contents, list_of_names, plotname)
        return children, fig.to_json()
    return None, None


@callback(
    Output("download-plot-png", "data"),
    Input("btn_png", "n_clicks"),
    State("plot-name", "value"),
    State("figure-store", "data"),
    prevent_initial_call=True,
)
def export_png(n_clicks, plotname, fig_data):
    fig = pio.from_json(fig_data)
    if n_clicks is not None:
        # Generate the PNG image of the figure
        fig_bytes = fig.to_image(format="png")

        # Encode the image bytes as base64
        base64_image = base64.b64encode(fig_bytes).decode("utf-8")

        # Prepare the data for download
        data = dict(
            content=base64_image, filename=f"{plotname}.png", base64=True, type="png"
        )

        return data
    return None


if __name__ == "__main__":
    app.run(debug=True)

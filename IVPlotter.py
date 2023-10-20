import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback, Output, Input, State
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import pandas as pd
import base64
import io
import names  # type: ignore
import re
# Initialize the app
app = dash.Dash()

app.layout = dbc.Container(
    [
        dbc.Container([html.Label(
            "Enter Plot Name:",
        ),
        dbc.Input(
            id="plot-name", type="text", value="My Plot", 
        ),],fluid=True, style={"padding": "10px"}),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "90%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "0 auto",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Checklist(
                    id="file-checklist",
                    style={"padding": "10px", 'fontSize': '0.8em'},
                ),width=3),
                dbc.Col(
                    id="output-data-upload",
                    width=9
                ),
            ],),
        html.Div(children=[dash_table.DataTable(
            id="Table",
            style_data={
                "whiteSpace": "normal",
            },
            style_cell={
                "whiteSpace": "normal",
                "font-size": "0.8rem",
                "overflowX": "clip",
                'maxWidth': '180px',
            },
            style_table={
                "overflow": "scroll",
                "display": "flex",
                'minWidth': '100%'
            },
            fixed_columns={"headers": True, "data": 1},
            style_header={
                'textAlign': 'center',
                'height': '4rem',
                "overflowX": "scroll",
                'minWidth': '0px', 'width':'150px' ,'maxWidth': '180px',
            }
        )], style={"display": "block", "width":"95%", "margin": "0 auto"},),
        dbc.ButtonGroup([
            dbc.Button("Download as png", id="btn_png"),
            dbc.Button("Download as html", id="btn_html"), 
        ], style={"padding": "10px"}),
        dcc.Download(id="download-plot-png"),
        dcc.Download(id="download-plot-html"),
        dcc.Store(id="figure-store", storage_type="memory"),
        dcc.Store(id="dataframe-store", storage_type="memory"),
    ], fluid=True
)


class IVdata:
    """
    A class to represent IV data.

    Attributes
    ----------
    data : pandas.DataFrame
        The IV data.
    metadata : pandas.DataFrame
        The metadata associated with the IV data.

    Methods
    -------
    get_data()
        Returns the IV data.
    get_metadata()
        Returns the metadata associated with the IV data.
    add(IVdat)
        Adds another IVdata object to the current object.
    to_json()
        Returns the IV data and metadata as a list of dictionaries.
    """

    def __init__(self, df1=pd.DataFrame(), df2=pd.DataFrame()):
        """
        Parameters
        ----------
        df1 : pandas.DataFrame, optional
            The IV data, by default an empty DataFrame.
        df2 : pandas.DataFrame, optional
            The metadata associated with the IV data, by default an empty DataFrame.
        """
        self.data = df1
        self.metadata = df2

    def get_data(self):
        """
        Returns
        -------
        pandas.DataFrame
            The IV data.
        """
        return self.data

    def get_metadata(self):
        """
        Returns
        -------
        pandas.DataFrame
            The metadata associated with the IV data.
        """
        return self.metadata

    def add(self, IVdat):
        """
        Adds another IVdata object to the current object.

        Parameters
        ----------
        IVdat : IVdata
            The IVdata object to add.
        """
        self.data = pd.concat([self.data, IVdat.data], ignore_index=True)
        self.metadata = pd.concat(
            [self.metadata, IVdat.metadata], ignore_index=True)

    def to_json(self):
        """
        Returns
        -------
        list
            The IV data and metadata as a list of dictionaries.
        """
        data = self.data
        datadict = data.to_dict()
        metadata = self.metadata
        metadatadict = metadata.to_dict()
        list = [datadict, metadatadict]
        return list


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return "Unset"

import hashlib
from io import StringIO

def create_unique_string(io_string):
    """
    Creates a unique string from an IOString.

    Args:
        io_string (io.StringIO): The IOString to create a unique string from.

    Returns:
        str: The unique string.
    """
    # Get the contents of the IOString
    contents = io_string.getvalue()

    # Create a hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the contents of the IOString
    hash_object.update(contents.encode())

    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()

    # Return the hexadecimal representation
    return hex_digest

def Hier(df):
    """
    Adds a new level to the index of a pandas DataFrame using groupby and cumcount, and then resets the index to keep only two levels.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.

    Returns:
        None
    """
    df.set_index([df.groupby(level=0).cumcount(), df.index],
                 append=True, inplace=True)

    df.reset_index(level=[2], drop=True, inplace=True)
    return


def json_to_IVdata(j):
    j1 = j[0]
    j2 = j[1]
    data = pd.DataFrame.from_dict(j1)
    metadata = pd.DataFrame.from_dict(j2)
    return IVdata(data, metadata)


def Readivdat(rawdata, mname):
    data = []
    START = False
    FOOT = False
    id = create_unique_string(rawdata)
    metadata = []
    mlabel = []
    DUTID = ""
    data_label = ""
    for line in rawdata:
        if "Date" in line:
            # Define a regular expression pattern to match the date portion
            pattern = r'Date:\s+"(.+)"'
            # Use re.search to find the date portion
            match = re.search(pattern, line)
            if match:
                date = match.group(1)  # Get the matched date portion
            else:
                date = "Date not found."
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
            input_power = float(
                line[output_index + len("Output(%):"): end_index])
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
    df_metadata["Name"] = mname + " | " + date  # Add Measurement Name
    df_metadata["id"] = id
    df_metadata["DUTID"] = DUTID
    df = pd.DataFrame(data, columns=data_label)
    df["Name"] = mname + " | " + date  # Add Measurement Name
    df["id"] = id
    DF = IVdata(df, df_metadata)
    return DF


def PlotIV(DF, title):
    """
    Plots current and power vs voltage for multiple measurements with different input powers.

    Args:
    - DF: a pandas DataFrame containing the data to plot and metadata about the measurements.
    - title: a string with the title of the plot.

    Returns:
    - fig: a plotly figure object with the current and power vs voltage plot.
    """

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
            # 'y':0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=50),
        },
    )

    return fig


def Process_Files(list_of_contents, list_of_names):
    d = IVdata()
    for content, mname in zip(list_of_contents, list_of_names):
        _, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        rawdata = io.StringIO(decoded.decode("utf-8"))
        data = Readivdat(rawdata, mname)
        data.data["Current(A)"] *= -1
        data.data["Power(W)"] *= -1
        d.add(data)
    return d


@callback(
    Output("dataframe-store", "data"),
    Output("file-checklist", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        DF = Process_Files(list_of_contents, list_of_names)
    A = DF.to_json()
    result = DF.data[["id", "Name"]].drop_duplicates()
    list_of_dicts = [
        {"label": row["Name"], "value": row["id"]} for _, row in result.iterrows()
    ]
    return A, list_of_dicts


@callback(
    Output("output-data-upload", "children"),
    Output("figure-store", "data"),
    Output("Table", "columns"),
    Output("Table", "data"),
    # Input("btn_plot", "n_clicks"),
    Input("file-checklist", "value"),
    State("plot-name", "value"),
    State("dataframe-store", "data"),
    prevent_initial_call=True,
)
def plot(value, plotname, dataframe_data):
    if value is not None:
        DF = json_to_IVdata(dataframe_data)
        DF.data = DF.data[DF.data["id"].isin(value)]
        DF.metadata = DF.metadata[DF.metadata["id"].isin(value)]
        DF.data.set_index("id", inplace=True)
        Hier(DF.data)
        fig = PlotIV(DF, plotname)
        DF.metadata = DF.metadata.set_index("Name").T.reset_index()
        # DF.metadata.index.name = None
        columns = [{"name": i, "id": i} for i in DF.metadata.columns]
        data = DF.metadata.to_dict("records")
        return html.Div([dcc.Graph(figure=fig)]), fig, columns, data


@callback(
    Output("download-plot-png", "data"),
    Input("btn_png", "n_clicks"),
    State("plot-name", "value"),
    State("figure-store", "data"),
    prevent_initial_call=True,
    allow_duplicates=True
)
def export_png(n_clicks, plotname, fig_data):
    fig = go.Figure(fig_data)
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


@callback(
    Output("download-plot-html", "data"),
    Input("btn_html", "n_clicks"),
    State("plot-name", "value"),
    State("figure-store", "data"),
    prevent_initial_call=True,
    allow_duplicates=True
)
def export_html(n_clicks, plotname, fig_data):
    fig = go.Figure(fig_data)
    if n_clicks is not None:
        # Generate the HTML file of the figure
        html_bytes = fig.to_html()

        # Prepare the data for download
        data = dict(
            content=html_bytes, filename=f"{plotname}.html", base64=False, type="html"
        )

        return data
    return None


if __name__ == "__main__":
    app.run(debug=True)

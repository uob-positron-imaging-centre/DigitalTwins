import numpy as np
import plotly.graph_objs as go
import seaborn as sns


# User defined variables
data_folder_path = 'data5'  # Path to folder containing data
file_name = 'bulk_density.npy'  # Name of file containing bulk density data

# Import data
bulk_density_data = np.load(f"{data_folder_path}/{file_name}")
all_taps = np.linspace(0, len(bulk_density_data), len(bulk_density_data)+1)
all_bulk_density = bulk_density_data

# Start graph
fig = go.Figure()
colours = sns.color_palette("colorblind")
colours = colours.as_hex()
fig.add_trace(go.Scatter(x=all_taps,
                         y=all_bulk_density,
                         mode='markers',  # or lines
                         marker=dict(size=8, color=colours[0]),  # Change colours value per trace added
                         )
              )

# Show graph
fig.update_layout(
    title_text="Bulk Density vs Tap Number",
    title_font_size=30,
    legend=dict(font=dict(size=20)))
fig.update_xaxes(
        title_text="Tap Number",
        title_font={"size": 20},
        tickfont_size=20)
fig.update_yaxes(
        title_text="Bulk Density (kg/m^3)",
        title_font={"size": 20},
        tickfont_size=20)
fig.show()

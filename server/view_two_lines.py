import pandas as pd
import folium

df = pd.read_csv("boundaries.csv")

# Allow either schema:
# - legacy: i,outer_lat,outer_lon,inner_lat,inner_lon
# - new:    time_s,outer_lat,outer_lon,inner_lat,inner_lon

center_lat = df["outer_lat"].mean()
center_lon = df["outer_lon"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=17)

outer_coords = list(zip(df["outer_lat"], df["outer_lon"]))
inner_coords = list(zip(df["inner_lat"], df["inner_lon"]))

folium.PolyLine(outer_coords, color="yellow", weight=4, tooltip="Outer").add_to(m)
folium.PolyLine(inner_coords, color="blue", weight=4, tooltip="Inner").add_to(m)

m.save("track_boundaries.html")
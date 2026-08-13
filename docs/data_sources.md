# Data sources

## Istanbul hourly traffic

Primary intended source: **Saatlik Trafik Yoğunluk Veri Seti / Hourly Traffic Density Data Set**, listed by the B40 Open Data Portal and linked to the Istanbul Metropolitan Municipality open-data platform.

B40 listing: `https://opendata.b40cities.org/tr/dataset/hourly-traffic-density-data-set`

IBB target page: `https://data.ibb.gov.tr/en/dataset/hourly-traffic-density-data-set`

The public B40 listing describes hourly Istanbul location, density, and traffic information. Because portal download endpoints and file packaging can change, this repository does not hard-code an unverified raw CSV URL. Download the current resource manually and record the file hash.

## Road topology

OpenStreetMap road networks may be accessed with OSMnx. OSMnx represents drivable street networks as directed MultiDiGraphs and preserves non-planar topology such as bridges/tunnels. OpenStreetMap data are subject to the ODbL and attribution requirements.

## Weather and events

These are planned exogenous layers, not required in v0.1.0. Any added source must receive a provenance entry before inclusion in confirmatory experiments.

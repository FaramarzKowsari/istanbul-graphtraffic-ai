# Data Card

## Intended dataset
Istanbul hourly traffic observations from the public IBB/B40 open-data listing.

## Redistribution
Raw third-party files are not included in this repository.

## Minimum required fields
`timestamp`, `sensor_id`, and `avg_speed` after schema adaptation.

## Strongly preferred fields
`latitude`, `longitude`, `vehicle_count`, `traffic_density`.

## Known risks
Sensor coverage may change over time; missing observations may not be random; public portal schemas may evolve; location identifiers may be geohashes or other spatial keys rather than permanent physical sensor IDs. These issues must be audited before scientific interpretation.

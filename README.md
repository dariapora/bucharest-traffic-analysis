# Bucharest Traffic Analysis

This project examines traffic flow across the Bucharest road network during August 25th-31st, 2024. The goal is to understand how traffic conditions vary across different road segments, times of day, and days of the week, providing a foundation for understanding urban mobility in Romania's capital.

## Dataset Overview

The dataset contains 10.4 million traffic observations distributed across 23,506 distinct street segments. Data was collected continuously at 15-minute intervals, capturing the complete temporal profile of traffic behavior throughout each day. The observations span one full week, including weekdays (Monday through Friday, August 26-30), weekend days (Saturday August 31 and Sunday August 25), providing temporal diversity in traffic patterns.

Each observation records multiple speed-related metrics for a specific road segment during a 15-minute window. These include median speed, average speed, harmonic average speed, and the full speed distribution expressed as percentiles. The data also captures variability through standard deviation of speeds and travel time measurements.

The Dataset contains data obtained through TomTom's [Trafic Stats API](https://www.tomtom.com/products/traffic-stats/).

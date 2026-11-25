# Raceline data

The TSPC node in this workspace does not load raceline CSV files from this
directory. It reads the selected map's planned trajectory from
`race_stack/stack_master/maps/<map_name>/global_waypoints.json`.

This directory is retained from the upstream TSPC project; no CSV file needs to
be added here for the simulator workflow documented in the parent README.

# Time the register-transpose graph and identify its waits

Take an event timestamp before and after each stage of all 128 products.
Report total and exclusive time for the input, middle and output streams.
Use the same timestamps to estimate when reuse of a forward or partial
slot is the last dependency holding up a stage. Count input waits when a
producer is the final dependency. Exclude the first product from these
wait totals to omit initial pipeline fill.

The wait totals describe separate streams and may overlap in time.
The profiler adds events to the graph, so compare its median with its
unprofiled source. Use the resulting waits to decide between fewer
slots, smaller producer windows and faster arithmetic. The timed
transform and all numerical inputs follow the unprofiled revision.
[Earlier timeline](../two-shortjointprofile/README.md)
